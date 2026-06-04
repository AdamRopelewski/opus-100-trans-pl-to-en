from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    gpu_name: str
    cuda_version: str
    memory_total_mb: int


class CudaRequiredError(RuntimeError):
    pass


def resolve_training_device(
    requested_device: str = "cuda",
    require_cuda: bool = True,
    allow_cpu_fallback: bool = False,
) -> DeviceInfo:
    if requested_device.startswith("cuda"):
        try:
            cuda_available = torch.cuda.is_available()
        except Exception as exc:  # pragma: no cover - depends on local driver state
            raise CudaRequiredError(f"CUDA initialization failed: {exc}") from exc
        if cuda_available:
            device = torch.device(requested_device)
            index = device.index if device.index is not None else 0
            props = torch.cuda.get_device_properties(index)
            return DeviceInfo(
                device=device,
                gpu_name=torch.cuda.get_device_name(index),
                cuda_version=str(torch.version.cuda or "unknown"),
                memory_total_mb=int(props.total_memory // (1024 * 1024)),
            )
        if require_cuda or not allow_cpu_fallback:
            raise CudaRequiredError(
                "CUDA is required for PyTorch training, but torch.cuda.is_available() is False. "
                "Check the NVIDIA driver and the installed PyTorch CUDA build."
            )

    if require_cuda and not allow_cpu_fallback:
        raise CudaRequiredError(f"CUDA is required for PyTorch training, but requested device was '{requested_device}'.")
    return DeviceInfo(device=torch.device("cpu"), gpu_name="cpu", cuda_version=str(torch.version.cuda or "none"), memory_total_mb=0)


def select_amp_precision(requested_precision: str, fallback_precision: str | None = "fp16") -> tuple[str, torch.dtype | None, bool]:
    precision = requested_precision.lower()
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16, False
        if fallback_precision:
            return select_amp_precision(fallback_precision, None)
        return "fp32", None, False
    if precision == "fp16":
        return "fp16", torch.float16, True
    if precision in {"fp32", "float32"}:
        return "fp32", None, False
    raise ValueError(f"Unsupported precision: {requested_precision}")
