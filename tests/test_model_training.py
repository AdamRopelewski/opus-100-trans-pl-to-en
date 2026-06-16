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
from scripts.train_model import _resolve_runtime_settings, _select_validation_dataset
from scripts.tatoeba.tokenize_splits import tokenize_split
from src.data.tokenized_translation_dataset import load_tokenized_translation_dataset
from src.data.translation_dataset import (
    SpecialTokenIds,
    SplitLoadStats,
    TranslationDataset,
    build_translation_examples,
    load_translation_dataset,
)
from src.model.transformer_nmt import TransformerNMT, TransformerNMTConfig
from src.train.device import (
    CudaRequiredError,
    resolve_training_device,
    select_amp_precision,
)
import src.train.model as model_module
from src.train.model import (
    LabelSmoothedCrossEntropy,
    ModelTrainConfig,
    TrainingComponents,
    build_inverse_sqrt_scheduler,
    load_model_checkpoint,
    train_model,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path), compression="zstd")


def _encode(text: str) -> list[int]:
    return [ord(ch) % 20 + 4 for ch in text.replace(" ", "")]


class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return _encode(text)


def _training_components(
    model: TransformerNMT,
    loader: torch.utils.data.DataLoader,
    criterion: LabelSmoothedCrossEntropy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> TrainingComponents:
    return TrainingComponents(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )


def test_dataset_encoding_places_bos_eos_and_drops_overlength() -> None:
    token_ids = SpecialTokenIds()
    rows = iter([("ab", "cd"), ("verylong", "x")])

    examples, rows_in, overlength = build_translation_examples(
        rows, _encode, token_ids, max_seq_len=4
    )

    assert rows_in == 2
    assert overlength == 1
    assert len(examples) == 1
    assert examples[0].src_ids[-1] == token_ids.eos_id
    assert examples[0].tgt_in_ids[0] == token_ids.bos_id
    assert examples[0].tgt_out_ids[-1] == token_ids.eos_id


def test_load_translation_dataset_supports_cleaned_parquet_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.parquet"
    _write_parquet(path, [{"pl": "kot", "en": "cat"}])

    dataset, stats = load_translation_dataset(
        "train", path, _encode, SpecialTokenIds(), max_seq_len=16
    )

    assert stats.rows_in == 1
    assert stats.rows_out == 1
    assert dataset[0].src_ids[-1] == 3


def test_tatoeba_tokenize_split_writes_tokenized_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "train.parquet"
    output_dir = tmp_path / "tokenized"
    _write_parquet(
        input_path,
        [
            {"pl": "kot", "en": "cat"},
            {"pl": "bardzo dlugie zdanie", "en": "long"},
        ],
    )

    stats = tokenize_split(
        "train",
        input_path,
        output_dir,
        _FakeTokenizer(),
        SpecialTokenIds(),
        max_seq_len=8,
        drop_overlength=True,
        shard_rows=1,
        overwrite=False,
        progress=False,
    )

    assert stats.rows_in == 2
    assert stats.rows_out == 1
    assert stats.overlength_rows == 1
    assert stats.files_written == 1
    table = pq.read_table(
        output_dir / "train" / "train-tokenized-00001-of-00001.parquet"
    )
    row = table.to_pylist()[0]
    assert row["src_ids"][-1] == SpecialTokenIds().eos_id
    assert "tgt_in_ids" not in row
    assert "tgt_out_ids" not in row
    assert row["tgt_ids"] == _encode("cat")


def test_tokenized_translation_dataset_streams_parquet(tmp_path: Path) -> None:
    path = tmp_path / "train-tokenized-00000.parquet"
    table = pa.Table.from_pylist(
        [
            {"src_ids": [4, 3], "tgt_ids": [5]},
            {"src_ids": [6, 3], "tgt_ids": [7]},
        ],
        schema=pa.schema(
            [
                pa.field("src_ids", pa.list_(pa.int32())),
                pa.field("tgt_ids", pa.list_(pa.int32())),
            ]
        ),
    )
    pq.write_table(table, path, compression="zstd")

    dataset, stats = load_tokenized_translation_dataset(
        "train", (path,), batch_size=1
    )
    examples = list(dataset)

    assert stats.rows_out == 2
    assert examples[0].src_ids == [4, 3]
    assert examples[0].tgt_in_ids == [SpecialTokenIds().bos_id, 5]
    assert examples[1].tgt_out_ids == [7, 3]


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

    assert mask.tolist() == [
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ]


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


def test_tied_decoder_embedding_init_keeps_logits_in_reasonable_range() -> None:
    torch.manual_seed(123)
    examples, _rows_in, _overlength = build_translation_examples(
        iter([("ab", "cd"), ("ef", "gh")]),
        _encode,
        SpecialTokenIds(),
        max_seq_len=8,
    )
    batch = TranslationCollator(pad_id=0, pad_to_multiple_of=4)(examples)
    model = TransformerNMT(
        TransformerNMTConfig(
            vocab_size=64,
            max_seq_len=8,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            tie_decoder_embeddings=True,
        )
    )
    criterion = LabelSmoothedCrossEntropy(label_smoothing=0.0, ignore_index=0)

    assert model.output_projection.weight is model.tgt_embedding.weight
    assert torch.equal(
        model.src_embedding.weight[0], torch.zeros_like(model.src_embedding.weight[0])
    )
    assert torch.equal(
        model.tgt_embedding.weight[0], torch.zeros_like(model.tgt_embedding.weight[0])
    )

    logits = model(
        batch.src_ids,
        batch.tgt_in_ids,
        batch.src_key_padding_mask,
        batch.tgt_key_padding_mask,
        batch.tgt_causal_mask,
    )
    loss = criterion(logits, batch.tgt_out_ids)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert float(logits.detach().abs().max()) < 20.0
    assert float(loss.detach()) < 20.0


def test_model_forward_and_backward() -> None:
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


def test_resolve_training_device_requires_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(CudaRequiredError, match="CUDA is required"):
        resolve_training_device("cuda", require_cuda=True, allow_cpu_fallback=False)


def test_resolve_training_device_returns_cuda_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Props:
        total_memory = 8 * 1024 * 1024

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: Props())
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "Test GPU")

    info = resolve_training_device("cuda", require_cuda=True, allow_cpu_fallback=False)

    assert info.device.type == "cuda"
    assert info.gpu_name == "Test GPU"
    assert info.memory_total_mb == 8


def test_select_amp_precision_falls_back_to_fp16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    name, dtype, use_scaler = select_amp_precision("bf16", "fp16")

    assert name == "fp16"
    assert dtype == torch.float16
    assert use_scaler


def test_overfit_runtime_forces_debug_settings() -> None:
    args = SimpleNamespace(
        overfit_samples=4, overfit_max_steps=200, overfit_loss_threshold=0.1
    )
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
    assert runtime.early_stopping_patience is None

    examples, _rows_in, _overlength = build_translation_examples(
        iter([("a", "x"), ("b", "y")]),
        _encode,
        SpecialTokenIds(),
        max_seq_len=8,
    )
    train_dataset = TranslationDataset(examples)
    train_stats = SplitLoadStats(
        split="train", rows_in=2, rows_out=2, overlength_rows=0
    )

    validation_dataset, validation_stats = _select_validation_dataset(
        runtime,
        train_dataset,
        train_stats,
        {},
        tokenizer=None,
        token_ids=SpecialTokenIds(),
        max_seq_len=8,
        drop_overlength=True,
    )

    assert validation_dataset is train_dataset
    assert validation_stats is train_stats


def test_full_runtime_reads_early_stopping_patience() -> None:
    args = SimpleNamespace(
        overfit_samples=None, overfit_max_steps=200, overfit_loss_threshold=0.1
    )
    runtime = _resolve_runtime_settings(
        {
            "stage5_model": {"preset": "small", "presets": {"small": {"dropout": 0.2}}},
            "stage6_train": {"early_stopping_patience": 7},
        },
        args,
    )

    assert runtime.mode == "full"
    assert runtime.early_stopping_patience == 7


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
    device_info = resolve_training_device(
        "cpu", require_cuda=False, allow_cpu_fallback=True
    )
    train_cfg = ModelTrainConfig(
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
        tensorboard_log_dir=tmp_path / "tensorboard",
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
    )

    components = _training_components(model, loader, criterion, optimizer, scheduler)
    best_loss = train_model(components, train_cfg, device_info)

    assert torch.isfinite(torch.tensor(best_loss))
    checkpoint = torch.load(train_cfg.best_checkpoint_path, map_location=device)
    assert checkpoint["global_step"] == 1
    assert checkpoint["best_validation_loss"] == checkpoint["validation_loss"]
    assert checkpoint["model_config"] == train_cfg.model_config
    assert checkpoint["tokenizer_path"] == "tokenizers/test.model"

    resumed_model = TransformerNMT(model_cfg)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = build_inverse_sqrt_scheduler(resumed_optimizer, warmup_steps=1)
    resume_state = load_model_checkpoint(
        train_cfg.best_checkpoint_path,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        device,
    )

    assert resume_state.global_step == 1
    assert resume_state.start_epoch == 2
    assert resume_state.best_validation_loss == checkpoint["best_validation_loss"]
    assert resume_state.learning_rates == pytest.approx(
        (checkpoint["scheduler_state_dict"]["_last_lr"][0],)
    )
    for original, resumed in zip(
        model.parameters(), resumed_model.parameters(), strict=True
    ):
        assert torch.equal(original, resumed)


def test_resume_without_scheduler_state_restores_lr_from_global_step(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_inverse_sqrt_scheduler(optimizer, warmup_steps=10)
    path = tmp_path / "old.pt"
    torch.save(
        {
            "epoch": 1,
            "global_step": 5,
            "best_validation_loss": 1.5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

    resumed_model = torch.nn.Linear(2, 1)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    resumed_scheduler = build_inverse_sqrt_scheduler(
        resumed_optimizer, warmup_steps=10
    )
    resume_state = load_model_checkpoint(
        path,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        torch.device("cpu"),
    )

    assert resume_state.global_step == 5
    assert resumed_scheduler.last_epoch == 5
    assert resumed_scheduler.get_last_lr() == pytest.approx([5e-4])
    assert resume_state.learning_rates == pytest.approx((5e-4,))


def test_train_model_stops_after_early_stopping_patience(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    examples, _rows_in, _overlength = build_translation_examples(
        iter([("ab", "cd"), ("ef", "gh"), ("ij", "kl"), ("mn", "op")]),
        _encode,
        SpecialTokenIds(),
        max_seq_len=8,
    )
    loader = torch.utils.data.DataLoader(
        TranslationDataset(examples),
        batch_size=1,
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
    device_info = resolve_training_device(
        "cpu", require_cuda=False, allow_cpu_fallback=True
    )
    validation_losses = iter([1.0, 1.1, 1.2])

    def fake_validate(*_args, **_kwargs) -> float:
        return next(validation_losses)

    monkeypatch.setattr(model_module, "validate", fake_validate)
    train_cfg = ModelTrainConfig(
        mode="full",
        num_epochs=5,
        grad_accum_steps=1,
        validate_every_steps=1,
        save_every_steps=100,
        grad_clip_norm=1.0,
        skip_nan_batches=False,
        max_steps=10,
        overfit_loss_threshold=None,
        last_checkpoint_path=tmp_path / "last.pt",
        best_checkpoint_path=tmp_path / "best.pt",
        tensorboard_log_dir=tmp_path / "tensorboard",
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
        early_stopping_patience=2,
    )

    components = _training_components(model, loader, criterion, optimizer, scheduler)
    best_loss = train_model(components, train_cfg, device_info)

    assert best_loss == 1.0
    assert train_cfg.last_checkpoint_path.exists()
    last_checkpoint = torch.load(
        train_cfg.last_checkpoint_path, map_location=torch.device("cpu")
    )
    assert last_checkpoint["global_step"] == 3
    assert last_checkpoint["best_validation_loss"] == 1.0
    assert any(train_cfg.tensorboard_log_dir.glob("events.out.tfevents.*"))
