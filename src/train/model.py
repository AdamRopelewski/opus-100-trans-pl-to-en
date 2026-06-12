from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.collate import TranslationBatch
from src.train.device import DeviceInfo


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, ignore_index: int = 0) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        return self.loss(logits.reshape(-1, vocab_size), targets.reshape(-1))


def build_inverse_sqrt_scheduler(optimizer: Optimizer, warmup_steps: int) -> LambdaLR:
    warmup = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        current_step = max(1, step)
        return min(current_step ** -0.5, current_step * (warmup ** -1.5)) * math.sqrt(warmup)

    return LambdaLR(optimizer, lr_lambda)


@dataclass(frozen=True)
class ModelTrainConfig:
    mode: str
    num_epochs: int
    grad_accum_steps: int
    validate_every_steps: int
    save_every_steps: int
    grad_clip_norm: float
    skip_nan_batches: bool
    max_steps: int | None
    overfit_loss_threshold: float | None
    last_checkpoint_path: Path
    best_checkpoint_path: Path
    log_jsonl_path: Path
    precision_name: str
    amp_dtype: torch.dtype | None
    use_grad_scaler: bool
    model_config: dict[str, Any]
    tokenizer_path: str
    tokenizer_special_ids: dict[str, int]
    vocab_size: int
    start_epoch: int = 1
    start_step: int = 0
    best_validation_loss: float = math.inf
    early_stopping_patience: int | None = None


@dataclass(frozen=True)
class ModelResumeState:
    start_epoch: int
    global_step: int
    best_validation_loss: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _checkpoint_config(config: ModelTrainConfig) -> dict:
    payload = asdict(config)
    payload["last_checkpoint_path"] = str(config.last_checkpoint_path)
    payload["best_checkpoint_path"] = str(config.best_checkpoint_path)
    payload["log_jsonl_path"] = str(config.log_jsonl_path)
    payload["amp_dtype"] = str(config.amp_dtype)
    return payload


def _amp_device_type(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _count_tokens(batch: TranslationBatch) -> int:
    src_tokens = int(batch.src_key_padding_mask.logical_not().sum().item())
    tgt_tokens = int(batch.tgt_out_ids.ne(0).sum().item())
    return src_tokens + tgt_tokens


def _config_with_best_loss(config: ModelTrainConfig, best_validation_loss: float) -> ModelTrainConfig:
    return ModelTrainConfig(**{**asdict(config), "best_validation_loss": best_validation_loss})


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    step: int,
    validation_loss: float | None,
    config: ModelTrainConfig,
    device_info: DeviceInfo,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "global_step": step,
            "validation_loss": validation_loss,
            "best_validation_loss": config.best_validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_config": _checkpoint_config(config),
            "model_config": config.model_config,
            "tokenizer_path": config.tokenizer_path,
            "tokenizer_special_ids": config.tokenizer_special_ids,
            "vocab_size": config.vocab_size,
            "device": str(device_info.device),
            "gpu_name": device_info.gpu_name,
            "created_at_utc": _utc_now(),
        },
        path,
    )


def load_model_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> ModelResumeState:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    global_step = int(checkpoint.get("global_step", checkpoint.get("step", 0)))
    epoch = int(checkpoint.get("epoch", 0))
    best_validation_loss = float(checkpoint.get("best_validation_loss", checkpoint.get("validation_loss", math.inf)))
    return ModelResumeState(
        start_epoch=epoch + 1,
        global_step=global_step,
        best_validation_loss=best_validation_loss,
    )


def validate(
    model: nn.Module,
    dataloader: DataLoader[TranslationBatch],
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            with autocast(device_type=_amp_device_type(device), dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(
                    batch.src_ids,
                    batch.tgt_in_ids,
                    batch.src_key_padding_mask,
                    batch.tgt_key_padding_mask,
                    batch.tgt_causal_mask,
                )
                loss = criterion(logits, batch.tgt_out_ids)
            total_loss += float(loss.item())
            total_batches += 1
    model.train()
    if total_batches == 0:
        return math.inf
    return total_loss / total_batches


def train_model(
    model: nn.Module,
    train_loader: DataLoader[TranslationBatch],
    validation_loader: DataLoader[TranslationBatch],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ModelTrainConfig,
    device_info: DeviceInfo,
) -> float:
    device = device_info.device
    model.to(device)
    scaler = GradScaler("cuda", enabled=config.use_grad_scaler)
    best_validation_loss = config.best_validation_loss
    validations_without_improvement = 0
    global_step = config.start_step
    optimizer.zero_grad(set_to_none=True)

    _write_log(
        config.log_jsonl_path,
        {
            "event": "start",
            "mode": config.mode,
            "time_utc": _utc_now(),
            "device": str(device),
            "gpu_name": device_info.gpu_name,
            "cuda_version": device_info.cuda_version,
            "memory_total_mb": device_info.memory_total_mb,
            "precision": config.precision_name,
            "start_epoch": config.start_epoch,
            "start_step": config.start_step,
            "best_validation_loss": best_validation_loss,
            "early_stopping_patience": config.early_stopping_patience,
        },
    )

    for epoch in range(config.start_epoch, config.num_epochs + 1):
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = batch.to(device)
            tokens_per_batch = _count_tokens(batch)
            with autocast(device_type=_amp_device_type(device), dtype=config.amp_dtype, enabled=config.amp_dtype is not None):
                logits = model(
                    batch.src_ids,
                    batch.tgt_in_ids,
                    batch.src_key_padding_mask,
                    batch.tgt_key_padding_mask,
                    batch.tgt_causal_mask,
                )
                loss = criterion(logits, batch.tgt_out_ids) / config.grad_accum_steps

            if not torch.isfinite(loss):
                if config.skip_nan_batches:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise FloatingPointError(f"Non-finite training loss at global step {global_step}: {loss.item()}")

            scaler.scale(loss).backward()
            if micro_step % config.grad_accum_steps != 0:
                continue

            grad_norm = None
            if config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm).item())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            train_loss = float(loss.item() * config.grad_accum_steps)
            current_lr = float(scheduler.get_last_lr()[0])
            should_validate = global_step == 1 or global_step % config.validate_every_steps == 0
            validation_loss = None
            early_stopping_triggered = False
            if should_validate:
                validation_loss = validate(model, validation_loader, criterion, device, config.amp_dtype)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    validations_without_improvement = 0
                    _save_checkpoint(
                        config.best_checkpoint_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        validation_loss,
                        _config_with_best_loss(config, best_validation_loss),
                        device_info,
                    )
                elif config.early_stopping_patience is not None:
                    validations_without_improvement += 1
                    early_stopping_triggered = validations_without_improvement >= config.early_stopping_patience

            if global_step % config.save_every_steps == 0 or should_validate:
                _save_checkpoint(
                    config.last_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    validation_loss,
                    _config_with_best_loss(config, best_validation_loss),
                    device_info,
                )

            _write_log(
                config.log_jsonl_path,
                {
                    "event": "step",
                    "mode": config.mode,
                    "time_utc": _utc_now(),
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "best_validation_loss": best_validation_loss,
                    "validations_without_improvement": validations_without_improvement,
                    "grad_norm": grad_norm,
                    "tokens_per_batch": tokens_per_batch,
                    "lr": current_lr,
                    "precision": config.precision_name,
                    "gpu_name": device_info.gpu_name,
                    "checkpoint": str(config.last_checkpoint_path),
                },
            )
            print(
                f"step={global_step} epoch={epoch} train_loss={train_loss:.4f} "
                f"validation_loss={validation_loss if validation_loss is not None else 'n/a'} "
                f"grad_norm={grad_norm if grad_norm is not None else 'n/a'} lr={current_lr:.6g}"
            )

            if early_stopping_triggered:
                checkpoint_config = _config_with_best_loss(config, best_validation_loss)
                _save_checkpoint(
                    config.last_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    validation_loss,
                    checkpoint_config,
                    device_info,
                )
                _write_log(
                    config.log_jsonl_path,
                    {
                        "event": "early_stop",
                        "mode": config.mode,
                        "time_utc": _utc_now(),
                        "epoch": epoch,
                        "step": global_step,
                        "validation_loss": validation_loss,
                        "best_validation_loss": best_validation_loss,
                        "validations_without_improvement": validations_without_improvement,
                        "early_stopping_patience": config.early_stopping_patience,
                        "checkpoint": str(config.last_checkpoint_path),
                    },
                )
                print(
                    "Early stopping triggered: "
                    f"{validations_without_improvement} validations without improvement "
                    f"at step={global_step}."
                )
                return best_validation_loss

            if (
                config.mode == "overfit"
                and config.overfit_loss_threshold is not None
                and validation_loss is not None
                and validation_loss <= config.overfit_loss_threshold
            ):
                return best_validation_loss

            if config.max_steps is not None and global_step >= config.max_steps:
                return best_validation_loss

    return best_validation_loss
