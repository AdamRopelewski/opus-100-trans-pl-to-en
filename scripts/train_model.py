from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.collate import TranslationCollator
from src.data.translation_dataset import SpecialTokenIds, load_translation_dataset
from src.model.transformer_nmt import TransformerNMT, TransformerNMTConfig, count_parameters
from src.train.device import CudaRequiredError, resolve_training_device, select_amp_precision
from src.train.model import (
    LabelSmoothedCrossEntropy,
    ModelResumeState,
    ModelTrainConfig,
    build_inverse_sqrt_scheduler,
    load_model_checkpoint,
    train_model,
)
from src.utils.config import get_nested, load_config


@dataclass(frozen=True)
class ModelRuntimeSettings:
    mode: str
    overfit_samples: int | None
    micro_batch_size: int
    grad_accum_steps: int
    num_epochs: int
    validate_every_steps: int
    save_every_steps: int
    max_steps: int | None
    overfit_loss_threshold: float | None
    early_stopping_patience: int | None
    dropout: float
    label_smoothing: float
    shuffle_train: bool
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    warmup_steps: int
    weight_decay: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PL->EN Transformer model on GPU.")
    parser.add_argument("--config", type=Path, default=Path("configs/project_config.yaml"), help="Path to project config YAML.")
    parser.add_argument("--overfit-samples", type=int, default=None, help="Train and validate on the first N train pairs.")
    parser.add_argument("--overfit-max-steps", type=int, default=2000, help="Optimizer-step limit for --overfit-samples.")
    parser.add_argument("--overfit-loss-threshold", type=float, default=0.1, help="Stop overfit mode once validation loss is at or below this value.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume model, optimizer, scheduler, epoch, and step from a checkpoint.")
    return parser


def _resolve_split_file(data_dir: Path, split_name: str, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one parquet for split '{split_name}' in '{data_dir}' using '{pattern}', found {len(matches)}."
        )
    return matches[0]


def _load_sentencepiece(model_path: Path):
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install sentencepiece to train model.") from exc
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    return processor


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_runtime_settings(config_data: dict, args: argparse.Namespace) -> ModelRuntimeSettings:
    overfit_enabled = args.overfit_samples is not None
    if overfit_enabled and args.overfit_samples <= 0:
        raise ValueError("--overfit-samples must be positive.")

    config_num_epochs = int(get_nested(config_data, "stage6_train.num_epochs", 12))
    smoke_enabled = bool(get_nested(config_data, "smoke.enabled", False))
    smoke_max_steps = int(get_nested(config_data, "smoke.max_steps", 20)) if smoke_enabled else None

    if overfit_enabled:
        overfit_samples = int(args.overfit_samples)
        max_steps = int(args.overfit_max_steps)
        if max_steps <= 0:
            raise ValueError("--overfit-max-steps must be positive.")
        return ModelRuntimeSettings(
            mode="overfit",
            overfit_samples=overfit_samples,
            micro_batch_size=min(32, overfit_samples),
            grad_accum_steps=1,
            num_epochs=max(config_num_epochs, max_steps),
            validate_every_steps=25,
            save_every_steps=25,
            max_steps=max_steps,
            overfit_loss_threshold=float(args.overfit_loss_threshold),
            early_stopping_patience=None,
            dropout=0.0,
            label_smoothing=0.0,
            shuffle_train=False,
            num_workers=0,
            pin_memory=False,
            prefetch_factor=0,
            warmup_steps=1,
            weight_decay=0.0,
        )

    preset_name = str(get_nested(config_data, "stage5_model.preset", "small"))
    early_stopping_patience = get_nested(config_data, "stage6_train.early_stopping_patience", 5)
    return ModelRuntimeSettings(
        mode="full",
        overfit_samples=None,
        micro_batch_size=int(get_nested(config_data, "stage6_train.micro_batch_size", 16)),
        grad_accum_steps=int(get_nested(config_data, "stage6_train.grad_accum_steps", 8)),
        num_epochs=config_num_epochs,
        validate_every_steps=int(get_nested(config_data, "stage6_train.validate_every_steps", 2000)),
        save_every_steps=int(get_nested(config_data, "stage6_train.save_every_steps", 2000)),
        max_steps=smoke_max_steps,
        overfit_loss_threshold=None,
        early_stopping_patience=None if early_stopping_patience is None else int(early_stopping_patience),
        dropout=float(get_nested(config_data, f"stage5_model.presets.{preset_name}.dropout", 0.1)),
        label_smoothing=float(get_nested(config_data, "stage6_train.label_smoothing", 0.1)),
        shuffle_train=bool(get_nested(config_data, "stage4_dataloader.shuffle_train", True)),
        num_workers=int(get_nested(config_data, "stage4_dataloader.num_workers", 4)),
        pin_memory=bool(get_nested(config_data, "stage4_dataloader.pin_memory", True)),
        prefetch_factor=int(get_nested(config_data, "stage4_dataloader.prefetch_factor", 2)),
        warmup_steps=int(get_nested(config_data, "stage6_train.warmup_steps", 4000)),
        weight_decay=float(get_nested(config_data, "stage6_train.weight_decay", 0.01)),
    )


def _model_config_from_runtime(config_data: dict, tokenizer, token_ids: SpecialTokenIds, max_seq_len: int, runtime: ModelRuntimeSettings) -> tuple[str, TransformerNMTConfig]:
    preset_name = str(get_nested(config_data, "stage5_model.preset", "small"))
    preset = get_nested(config_data, f"stage5_model.presets.{preset_name}", {})
    return (
        preset_name,
        TransformerNMTConfig(
            vocab_size=int(tokenizer.vocab_size()),
            pad_id=token_ids.pad_id,
            max_seq_len=max_seq_len,
            d_model=int(preset.get("d_model", 256)),
            nhead=int(preset.get("nhead", 8)),
            num_encoder_layers=int(preset.get("num_encoder_layers", 4)),
            num_decoder_layers=int(preset.get("num_decoder_layers", 4)),
            dim_feedforward=int(preset.get("dim_feedforward", 1024)),
            dropout=runtime.dropout,
            tie_decoder_embeddings=bool(preset.get("tie_decoder_embeddings", True)),
        ),
    )


def _loader_kwargs(runtime: ModelRuntimeSettings, collator: TranslationCollator) -> dict:
    kwargs = {
        "batch_size": runtime.micro_batch_size,
        "num_workers": runtime.num_workers,
        "pin_memory": runtime.pin_memory,
        "collate_fn": collator,
    }
    if runtime.num_workers > 0:
        kwargs["prefetch_factor"] = runtime.prefetch_factor
    return kwargs


def _select_validation_dataset(
    runtime: ModelRuntimeSettings,
    train_dataset,
    train_stats,
    split_files: dict[str, Path],
    tokenizer,
    token_ids: SpecialTokenIds,
    max_seq_len: int,
    drop_overlength: bool,
    limit_validation: int | None,
):
    if runtime.mode == "overfit":
        return train_dataset, train_stats
    return load_translation_dataset(
        "validation",
        split_files["validation"],
        tokenizer.encode,
        token_ids,
        max_seq_len=max_seq_len,
        drop_overlength=drop_overlength,
        limit_samples=limit_validation,
    )


def main() -> int:
    args = _build_parser().parse_args()
    config_data = load_config(args.config)

    if not bool(get_nested(config_data, "stage4_dataloader.enabled", True)):
        print("stage4_dataloader.enabled is false, skipping model training.")
        return 0
    if not bool(get_nested(config_data, "stage5_model.enabled", True)):
        print("stage5_model.enabled is false; model training needs a model config.", file=sys.stderr)
        return 2
    if not bool(get_nested(config_data, "stage6_train.enabled", True)):
        print("stage6_train.enabled is false; model training needs train settings.", file=sys.stderr)
        return 2

    try:
        runtime = _resolve_runtime_settings(config_data, args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        device_info = resolve_training_device(
            requested_device=str(get_nested(config_data, "stage6_train.device", "cuda")),
            require_cuda=bool(get_nested(config_data, "stage6_train.require_cuda", True)),
            allow_cpu_fallback=bool(get_nested(config_data, "stage6_train.allow_cpu_fallback", False)),
        )
    except CudaRequiredError as exc:
        print(str(exc), file=sys.stderr)
        return 6

    seed = int(get_nested(config_data, "project.seed", 42))
    _seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    requested_precision = str(get_nested(config_data, "stage6_train.precision", "bf16"))
    fallback_precision = str(get_nested(config_data, "stage6_train.fallback_precision", "fp16"))
    precision_name, amp_dtype, use_grad_scaler = select_amp_precision(requested_precision, fallback_precision)
    print(f"Using device: {device_info.device}")
    print(f"GPU: {device_info.gpu_name}")
    print(f"CUDA runtime: {device_info.cuda_version}")
    print(f"GPU memory: {device_info.memory_total_mb} MB")
    print(f"Precision: {precision_name}")
    print(f"Training mode: {runtime.mode}")

    processed_dir = Path(get_nested(config_data, "stage2_cleaning.outputs.processed_dir", "data/processed/en-pl"))
    split_patterns = {
        "train": str(get_nested(config_data, "dataset.splits.train_pattern", "train-*.parquet")),
        "validation": str(get_nested(config_data, "dataset.splits.validation_pattern", "validation-*.parquet")),
        "test": str(get_nested(config_data, "dataset.splits.test_pattern", "test-*.parquet")),
    }
    try:
        split_files = {
            split: _resolve_split_file(processed_dir, split, split_patterns[split])
            for split in ("train", "validation", "test")
        }
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    tokenizer_prefix = Path(get_nested(config_data, "stage3_tokenizer.model_prefix", "tokenizers/spm_pl_en"))
    tokenizer_path = tokenizer_prefix.with_suffix(".model")
    tokenizer = _load_sentencepiece(tokenizer_path)
    token_ids = SpecialTokenIds(
        pad_id=int(tokenizer.pad_id()),
        unk_id=int(tokenizer.unk_id()),
        bos_id=int(tokenizer.bos_id()),
        eos_id=int(tokenizer.eos_id()),
    )
    if token_ids != SpecialTokenIds():
        print(f"Unexpected tokenizer special token ids: {token_ids}", file=sys.stderr)
        return 4

    smoke_enabled = bool(get_nested(config_data, "smoke.enabled", False))
    limit_train = runtime.overfit_samples
    if limit_train is None and smoke_enabled:
        limit_train = int(get_nested(config_data, "smoke.limit_train_samples", 1024))
    limit_validation = None
    if runtime.mode == "full" and smoke_enabled:
        limit_validation = int(get_nested(config_data, "smoke.limit_validation_samples", 256))

    max_seq_len = int(get_nested(config_data, "stage4_dataloader.max_seq_len", 128))
    drop_overlength = bool(get_nested(config_data, "stage4_dataloader.drop_overlength", True))
    train_dataset, train_stats = load_translation_dataset(
        "train",
        split_files["train"],
        tokenizer.encode,
        token_ids,
        max_seq_len=max_seq_len,
        drop_overlength=drop_overlength,
        limit_samples=limit_train,
    )
    validation_dataset, validation_stats = _select_validation_dataset(
        runtime,
        train_dataset,
        train_stats,
        split_files,
        tokenizer,
        token_ids,
        max_seq_len,
        drop_overlength,
        limit_validation,
    )
    test_dataset, test_stats = load_translation_dataset(
        "test",
        split_files["test"],
        tokenizer.encode,
        token_ids,
        max_seq_len=max_seq_len,
        drop_overlength=drop_overlength,
        limit_samples=1,
    )
    if len(train_dataset) == 0 or len(validation_dataset) == 0 or len(test_dataset) == 0:
        print(f"Empty model dataset after filtering: {train_stats}, {validation_stats}, {test_stats}", file=sys.stderr)
        return 5
    print(f"Loaded train: {train_stats.rows_out}/{train_stats.rows_in} rows; overlength={train_stats.overlength_rows}")
    print(
        f"Loaded validation: {validation_stats.rows_out}/{validation_stats.rows_in} rows; "
        f"overlength={validation_stats.overlength_rows}"
    )
    print(f"Verified test batching sample: {test_stats.rows_out}/{test_stats.rows_in} rows")

    collator = TranslationCollator(
        pad_id=token_ids.pad_id,
        pad_to_multiple_of=int(get_nested(config_data, "stage4_dataloader.pad_to_multiple_of", 8)),
    )
    loader_kwargs = _loader_kwargs(runtime, collator)
    train_loader = DataLoader(train_dataset, shuffle=runtime.shuffle_train, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)

    preset_name, model_cfg = _model_config_from_runtime(config_data, tokenizer, token_ids, max_seq_len, runtime)
    model = TransformerNMT(model_cfg)
    print(f"Model preset: {preset_name}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    if runtime.mode == "overfit":
        print(
            "Overfit debug: dropout=0.0 label_smoothing=0.0 shuffle=False "
            f"num_workers=0 batch_size={runtime.micro_batch_size}"
        )

    criterion = LabelSmoothedCrossEntropy(label_smoothing=runtime.label_smoothing, ignore_index=token_ids.pad_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(get_nested(config_data, "stage6_train.lr_peak", 3.0e-4)),
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=runtime.weight_decay,
    )
    scheduler = build_inverse_sqrt_scheduler(optimizer, runtime.warmup_steps)

    resume_state = ModelResumeState(start_epoch=1, global_step=0, best_validation_loss=float("inf"))
    if args.resume is not None:
        try:
            resume_state = load_model_checkpoint(args.resume, model, optimizer, scheduler, device_info.device)
        except Exception as exc:
            print(f"Failed to resume from {args.resume}: {exc}", file=sys.stderr)
            return 7
        print(
            f"Resumed from {args.resume}: start_epoch={resume_state.start_epoch} "
            f"global_step={resume_state.global_step} best_validation_loss={resume_state.best_validation_loss}"
        )

    token_id_payload = {
        "pad_id": token_ids.pad_id,
        "unk_id": token_ids.unk_id,
        "bos_id": token_ids.bos_id,
        "eos_id": token_ids.eos_id,
    }
    train_cfg = ModelTrainConfig(
        mode=runtime.mode,
        num_epochs=runtime.num_epochs,
        grad_accum_steps=runtime.grad_accum_steps,
        validate_every_steps=runtime.validate_every_steps,
        save_every_steps=runtime.save_every_steps,
        grad_clip_norm=float(get_nested(config_data, "stage6_train.grad_clip_norm", 1.0)),
        skip_nan_batches=bool(get_nested(config_data, "stage6_train.skip_nan_batches", True)),
        max_steps=runtime.max_steps,
        overfit_loss_threshold=runtime.overfit_loss_threshold,
        early_stopping_patience=runtime.early_stopping_patience,
        last_checkpoint_path=Path(get_nested(config_data, "stage6_train.output_last_checkpoint", "checkpoints/model_last.pt")),
        best_checkpoint_path=Path(get_nested(config_data, "stage6_train.output_best_checkpoint", "checkpoints/model_best.pt")),
        log_jsonl_path=Path(get_nested(config_data, "stage6_train.log_jsonl", "logs/model_train.jsonl")),
        precision_name=precision_name,
        amp_dtype=amp_dtype,
        use_grad_scaler=use_grad_scaler,
        model_config=asdict(model_cfg),
        tokenizer_path=str(tokenizer_path),
        tokenizer_special_ids=token_id_payload,
        vocab_size=int(tokenizer.vocab_size()),
        start_epoch=resume_state.start_epoch,
        start_step=resume_state.global_step,
        best_validation_loss=resume_state.best_validation_loss,
    )

    best_loss = train_model(model, train_loader, validation_loader, criterion, optimizer, scheduler, train_cfg, device_info)
    print(f"Model training complete. Best validation loss: {best_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
