from __future__ import annotations

import json
import math
from dataclasses import dataclass
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
        self.loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=ignore_index
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        return self.loss(logits.reshape(-1, vocab_size), targets.reshape(-1))


def build_inverse_sqrt_scheduler(optimizer: Optimizer, warmup_steps: int) -> LambdaLR:
    warmup = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        current_step = max(1, step)
        return min(current_step**-0.5, current_step * (warmup**-1.5)) * math.sqrt(
            warmup
        )

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
    amp_dtype: torch.dtype | None
    use_grad_scaler: bool
    model_config: dict[str, Any]
    tokenizer_path: str
    start_epoch: int = 1
    start_step: int = 0
    best_validation_loss: float = math.inf
    early_stopping_patience: int | None = None


@dataclass(frozen=True)
class ModelResumeState:
    start_epoch: int
    global_step: int
    best_validation_loss: float


@dataclass(frozen=True)
class TrainingComponents:
    model: nn.Module
    train_loader: DataLoader[TranslationBatch]
    validation_loader: DataLoader[TranslationBatch]
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


@dataclass
class TrainingState:
    epoch: int
    global_step: int
    best_validation_loss: float
    validations_without_improvement: int = 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _amp_device_type(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    step: int,
    validation_loss: float | None,
    config: ModelTrainConfig,
    best_validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "validation_loss": validation_loss,
            "best_validation_loss": best_validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": config.model_config,
            "tokenizer_path": config.tokenizer_path,
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
    best_validation_loss = float(
        checkpoint.get(
            "best_validation_loss", checkpoint.get("validation_loss", math.inf)
        )
    )
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
            with autocast(
                device_type=_amp_device_type(device),
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
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


class TrainingSession:
    def __init__(
        self,
        components: TrainingComponents,
        config: ModelTrainConfig,
        device_info: DeviceInfo,
    ) -> None:
        self.components = components
        self.config = config
        self.device_info = device_info
        self.device = device_info.device
        self.scaler = GradScaler("cuda", enabled=config.use_grad_scaler)
        self.state = TrainingState(
            epoch=config.start_epoch,
            global_step=config.start_step,
            best_validation_loss=config.best_validation_loss,
        )

    def run(self) -> float:
        self.components.model.to(self.device)
        self.components.optimizer.zero_grad(set_to_none=True)
        self._write_start_log()

        for epoch in range(self.config.start_epoch, self.config.num_epochs + 1):
            self.state.epoch = epoch
            if self._run_epoch():
                break
        return self.state.best_validation_loss

    def _run_epoch(self) -> bool:
        for micro_step, batch in enumerate(self.components.train_loader, start=1):
            batch = batch.to(self.device)
            loss = self._batch_loss(batch)

            if not torch.isfinite(loss):
                if self.config.skip_nan_batches:
                    self.components.optimizer.zero_grad(set_to_none=True)
                    continue
                raise FloatingPointError(
                    f"Non-finite training loss at global step {self.state.global_step}: {loss.item()}"
                )

            self.scaler.scale(loss).backward()
            if micro_step % self.config.grad_accum_steps != 0:
                continue

            grad_norm = self._finish_optimizer_step()
            self.state.global_step += 1

            train_loss = float(loss.item() * self.config.grad_accum_steps)
            validation_loss, early_stopping_triggered = self._validate_if_needed()
            self._save_last_checkpoint_if_needed(validation_loss)
            self._log_and_print_step(train_loss, validation_loss, grad_norm)

            if early_stopping_triggered:
                self._stop_after_early_stopping(validation_loss)
                return True
            if (
                self._reached_overfit_target(validation_loss)
                or self._reached_max_steps()
            ):
                return True
        return False

    def _batch_loss(self, batch: TranslationBatch) -> torch.Tensor:
        with autocast(
            device_type=_amp_device_type(self.device),
            dtype=self.config.amp_dtype,
            enabled=self.config.amp_dtype is not None,
        ):
            logits = self.components.model(
                batch.src_ids,
                batch.tgt_in_ids,
                batch.src_key_padding_mask,
                batch.tgt_key_padding_mask,
                batch.tgt_causal_mask,
            )
            return (
                self.components.criterion(logits, batch.tgt_out_ids)
                / self.config.grad_accum_steps
            )

    def _finish_optimizer_step(self) -> float | None:
        grad_norm = None
        if self.config.grad_clip_norm > 0:
            self.scaler.unscale_(self.components.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.components.model.parameters(),
                    self.config.grad_clip_norm,
                ).item()
            )
        self.scaler.step(self.components.optimizer)
        self.scaler.update()
        self.components.scheduler.step()
        self.components.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    def _validate_if_needed(self) -> tuple[float | None, bool]:
        if not self._should_validate():
            return None, False

        validation_loss = validate(
            self.components.model,
            self.components.validation_loader,
            self.components.criterion,
            self.device,
            self.config.amp_dtype,
        )
        if validation_loss < self.state.best_validation_loss:
            self.state.best_validation_loss = validation_loss
            self.state.validations_without_improvement = 0
            self._save_best_checkpoint(validation_loss)
            return validation_loss, False

        if self.config.early_stopping_patience is None:
            return validation_loss, False
        self.state.validations_without_improvement += 1
        should_stop = (
            self.state.validations_without_improvement
            >= self.config.early_stopping_patience
        )
        return validation_loss, should_stop

    def _should_validate(self) -> bool:
        return (
            self.state.global_step == 1
            or self.state.global_step % self.config.validate_every_steps == 0
        )

    def _save_best_checkpoint(self, validation_loss: float) -> None:
        self._save_checkpoint(self.config.best_checkpoint_path, validation_loss)

    def _save_last_checkpoint_if_needed(self, validation_loss: float | None) -> None:
        if (
            self.state.global_step % self.config.save_every_steps == 0
            or self._should_validate()
        ):
            self._save_checkpoint(self.config.last_checkpoint_path, validation_loss)

    def _save_checkpoint(self, path: Path, validation_loss: float | None) -> None:
        _save_checkpoint(
            path,
            self.components.model,
            self.components.optimizer,
            self.components.scheduler,
            self.state.epoch,
            self.state.global_step,
            validation_loss,
            self.config,
            self.state.best_validation_loss,
        )

    def _write_start_log(self) -> None:
        _write_log(
            self.config.log_jsonl_path,
            {
                "event": "start",
                "mode": self.config.mode,
                "time_utc": _utc_now(),
                "start_step": self.config.start_step,
            },
        )

    def _log_and_print_step(
        self,
        train_loss: float,
        validation_loss: float | None,
        grad_norm: float | None,
    ) -> None:
        current_lr = float(self.components.scheduler.get_last_lr()[0])
        self._write_step_log(train_loss, validation_loss, grad_norm, current_lr)
        print(
            f"step={self.state.global_step} epoch={self.state.epoch} train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss if validation_loss is not None else 'n/a'} "
            f"grad_norm={grad_norm if grad_norm is not None else 'n/a'} lr={current_lr:.6g}"
        )

    def _write_step_log(
        self,
        train_loss: float,
        validation_loss: float | None,
        grad_norm: float | None,
        current_lr: float,
    ) -> None:
        _write_log(
            self.config.log_jsonl_path,
            {
                "event": "step",
                "mode": self.config.mode,
                "time_utc": _utc_now(),
                "epoch": self.state.epoch,
                "step": self.state.global_step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "best_validation_loss": self.state.best_validation_loss,
                "grad_norm": grad_norm,
                "lr": current_lr,
            },
        )

    def _stop_after_early_stopping(self, validation_loss: float | None) -> None:
        self._save_checkpoint(self.config.last_checkpoint_path, validation_loss)
        self._write_early_stop_log(validation_loss)
        print(
            "Early stopping triggered: "
            f"{self.state.validations_without_improvement} validations without improvement "
            f"at step={self.state.global_step}."
        )

    def _write_early_stop_log(self, validation_loss: float | None) -> None:
        _write_log(
            self.config.log_jsonl_path,
            {
                "event": "early_stop",
                "mode": self.config.mode,
                "time_utc": _utc_now(),
                "epoch": self.state.epoch,
                "step": self.state.global_step,
                "validation_loss": validation_loss,
                "best_validation_loss": self.state.best_validation_loss,
            },
        )

    def _reached_overfit_target(self, validation_loss: float | None) -> bool:
        return (
            self.config.mode == "overfit"
            and self.config.overfit_loss_threshold is not None
            and validation_loss is not None
            and validation_loss <= self.config.overfit_loss_threshold
        )

    def _reached_max_steps(self) -> bool:
        return (
            self.config.max_steps is not None
            and self.state.global_step >= self.config.max_steps
        )


def train_model(
    components: TrainingComponents, config: ModelTrainConfig, device_info: DeviceInfo
) -> float:
    return TrainingSession(components, config, device_info).run()
