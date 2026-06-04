from __future__ import annotations

from pathlib import Path
import sys
import warnings
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.data.collate import TranslationCollator, make_causal_mask
from scripts.train_stage4 import _resolve_runtime_settings, _select_validation_dataset
from src.data.translation_dataset import (
    SpecialTokenIds,
    SplitLoadStats,
    TranslationDataset,
    build_translation_examples,
    load_translation_dataset,
)
from src.model.transformer_nmt import TransformerNMT, TransformerNMTConfig
from src.train.device import CudaRequiredError, resolve_training_device, select_amp_precision
from src.train.losses import LabelSmoothedCrossEntropy
from src.train.scheduler import build_inverse_sqrt_scheduler
from src.train.stage4 import Stage4TrainConfig, load_stage4_checkpoint, train_stage4


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path), compression="zstd")


def _encode(text: str) -> list[int]:
    return [ord(ch) % 20 + 4 for ch in text.replace(" ", "")]


def test_dataset_encoding_places_bos_eos_and_drops_overlength() -> None:
    token_ids = SpecialTokenIds()
    rows = iter([("ab", "cd"), ("verylong", "x")])

    examples, rows_in, overlength = build_translation_examples(rows, _encode, token_ids, max_seq_len=4)

    assert rows_in == 2
    assert overlength == 1
    assert len(examples) == 1
    assert examples[0].src_ids[-1] == token_ids.eos_id
    assert examples[0].tgt_in_ids[0] == token_ids.bos_id
    assert examples[0].tgt_out_ids[-1] == token_ids.eos_id


def test_load_translation_dataset_supports_cleaned_parquet_columns(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    _write_parquet(path, [{"pl": "kot", "en": "cat"}])

    dataset, stats = load_translation_dataset("train", path, _encode, SpecialTokenIds(), max_seq_len=16)

    assert stats.rows_in == 1
    assert stats.rows_out == 1
    assert dataset[0].src_ids[-1] == 3


def test_collator_pads_masks_and_causal_mask() -> None:
    examples, _rows_in, _overlength = build_translation_examples(
        iter([("ab", "c"), ("a", "cde")]),
        _encode,
        SpecialTokenIds(),
        max_seq_len=16,
    )
    batch = TranslationCollator(pad_id=0, pad_to_multiple_of=4)(examples)

    assert batch.src_ids.shape == (2, 4)
    assert batch.tgt_in_ids.shape == (2, 4)
    assert batch.src_key_padding_mask.dtype == torch.bool
    assert batch.src_key_padding_mask[1, -1]
    assert batch.tgt_causal_mask[0, 1]
    assert not batch.tgt_causal_mask[1, 0]


def test_make_causal_mask_blocks_future_tokens() -> None:
    mask = make_causal_mask(3)

    assert mask.tolist() == [[False, True, True], [False, False, True], [False, False, False]]


def test_transformer_nmt_constructs_without_nested_tensor_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TransformerNMT(
            TransformerNMTConfig(
                vocab_size=32,
                max_seq_len=8,
                d_model=16,
                nhead=4,
                num_encoder_layers=1,
                num_decoder_layers=1,
                dim_feedforward=32,
                dropout=0.0,
            )
        )

    messages = [str(warning.message) for warning in caught]
    assert not any("enable_nested_tensor is True" in message for message in messages)


def test_model_forward_and_backward_smoke() -> None:
    examples, _rows_in, _overlength = build_translation_examples(
        iter([("ab", "cd"), ("ef", "gh")]),
        _encode,
        SpecialTokenIds(),
        max_seq_len=8,
    )
    batch = TranslationCollator(pad_id=0, pad_to_multiple_of=4)(examples)
    model = TransformerNMT(
        TransformerNMTConfig(
            vocab_size=32,
            max_seq_len=8,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
    )
    criterion = LabelSmoothedCrossEntropy(label_smoothing=0.0, ignore_index=0)

    logits = model(
        batch.src_ids,
        batch.tgt_in_ids,
        batch.src_key_padding_mask,
        batch.tgt_key_padding_mask,
        batch.tgt_causal_mask,
    )
    loss = criterion(logits, batch.tgt_out_ids)
    loss.backward()

    assert logits.shape == (2, 4, 32)
    assert torch.isfinite(loss)


def test_resolve_training_device_requires_cuda_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(CudaRequiredError, match="CUDA is required"):
        resolve_training_device("cuda", require_cuda=True, allow_cpu_fallback=False)


def test_resolve_training_device_returns_cuda_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    class Props:
        total_memory = 8 * 1024 * 1024

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: Props())
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")

    info = resolve_training_device("cuda", require_cuda=True, allow_cpu_fallback=False)

    assert info.device.type == "cuda"
    assert info.gpu_name == "Test GPU"
    assert info.memory_total_mb == 8


def test_select_amp_precision_falls_back_to_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    name, dtype, use_scaler = select_amp_precision("bf16", "fp16")

    assert name == "fp16"
    assert dtype == torch.float16
    assert use_scaler


def test_overfit_runtime_forces_debug_settings() -> None:
    args = SimpleNamespace(overfit_samples=4, overfit_max_steps=200, overfit_loss_threshold=0.1)
    runtime = _resolve_runtime_settings({"stage6_train": {"num_epochs": 12}}, args)

    assert runtime.mode == "overfit"
    assert runtime.micro_batch_size == 4
    assert runtime.grad_accum_steps == 1
    assert runtime.num_epochs == 200
    assert runtime.validate_every_steps == 25
    assert runtime.dropout == 0.0
    assert runtime.label_smoothing == 0.0
    assert not runtime.shuffle_train
    assert runtime.num_workers == 0
    assert not runtime.pin_memory
    assert runtime.warmup_steps == 1
    assert runtime.weight_decay == 0.0

    examples, _rows_in, _overlength = build_translation_examples(
        iter([("a", "x"), ("b", "y")]),
        _encode,
        SpecialTokenIds(),
        max_seq_len=8,
    )
    train_dataset = TranslationDataset(examples)
    train_stats = SplitLoadStats(split="train", rows_in=2, rows_out=2, overlength_rows=0)

    validation_dataset, validation_stats = _select_validation_dataset(
        runtime,
        train_dataset,
        train_stats,
        {},
        tokenizer=None,
        token_ids=SpecialTokenIds(),
        max_seq_len=8,
        drop_overlength=True,
        limit_validation=None,
    )

    assert validation_dataset is train_dataset
    assert validation_stats is train_stats


def test_checkpoint_metadata_and_resume_state(tmp_path: Path) -> None:
    examples, _rows_in, _overlength = build_translation_examples(
        iter([("ab", "cd"), ("ef", "gh")]),
        _encode,
        SpecialTokenIds(),
        max_seq_len=8,
    )
    loader = torch.utils.data.DataLoader(
        TranslationDataset(examples),
        batch_size=2,
        shuffle=False,
        collate_fn=TranslationCollator(pad_id=0, pad_to_multiple_of=4),
    )
    model_cfg = TransformerNMTConfig(
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )
    model = TransformerNMT(model_cfg)
    criterion = LabelSmoothedCrossEntropy(label_smoothing=0.0, ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_inverse_sqrt_scheduler(optimizer, warmup_steps=1)
    device = torch.device("cpu")
    device_info = resolve_training_device("cpu", require_cuda=False, allow_cpu_fallback=True)
    train_cfg = Stage4TrainConfig(
        mode="overfit",
        num_epochs=1,
        grad_accum_steps=1,
        validate_every_steps=1,
        save_every_steps=1,
        grad_clip_norm=1.0,
        skip_nan_batches=False,
        max_steps=1,
        overfit_loss_threshold=None,
        last_checkpoint_path=tmp_path / "last.pt",
        best_checkpoint_path=tmp_path / "best.pt",
        log_jsonl_path=tmp_path / "train.jsonl",
        precision_name="fp32",
        amp_dtype=None,
        use_grad_scaler=False,
        model_config={
            "vocab_size": 32,
            "pad_id": 0,
            "max_seq_len": 8,
            "d_model": 16,
            "nhead": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "tie_decoder_embeddings": True,
        },
        tokenizer_path="tokenizers/test.model",
        tokenizer_special_ids={"pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3},
        vocab_size=32,
    )

    best_loss = train_stage4(model, loader, loader, criterion, optimizer, scheduler, train_cfg, device_info)

    assert torch.isfinite(torch.tensor(best_loss))
    checkpoint = torch.load(train_cfg.best_checkpoint_path, map_location=device)
    assert checkpoint["global_step"] == 1
    assert checkpoint["best_validation_loss"] == checkpoint["validation_loss"]
    assert checkpoint["model_config"] == train_cfg.model_config
    assert checkpoint["tokenizer_path"] == "tokenizers/test.model"
    assert checkpoint["tokenizer_special_ids"] == {"pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3}
    assert checkpoint["vocab_size"] == 32

    resumed_model = TransformerNMT(model_cfg)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = build_inverse_sqrt_scheduler(resumed_optimizer, warmup_steps=1)
    resume_state = load_stage4_checkpoint(
        train_cfg.best_checkpoint_path,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        device,
    )

    assert resume_state.global_step == 1
    assert resume_state.start_epoch == 2
    assert resume_state.best_validation_loss == checkpoint["best_validation_loss"]
    for original, resumed in zip(model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(original, resumed)
