from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.export_inference_checkpoint import (
    export_inference_checkpoint,
    strip_training_state,
)
from scripts.translate import _checkpoint_path


def test_strip_training_state_removes_optimizer_and_scheduler() -> None:
    checkpoint = {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "model_config": {"vocab_size": 8},
        "optimizer_state_dict": {"state": {}},
        "scheduler_state_dict": {"last_epoch": 1},
        "tokenizer_path": "tokenizers/test.model",
    }

    stripped = strip_training_state(checkpoint)

    assert "optimizer_state_dict" not in stripped
    assert "scheduler_state_dict" not in stripped
    assert stripped["model_state_dict"] == checkpoint["model_state_dict"]
    assert stripped["model_config"] == checkpoint["model_config"]
    assert stripped["checkpoint_type"] == "inference"
    assert stripped["stripped_keys"] == ["optimizer_state_dict", "scheduler_state_dict"]


def test_strip_training_state_requires_model_payload() -> None:
    with pytest.raises(ValueError, match="model_state_dict"):
        strip_training_state({"model_config": {}})


def test_export_inference_checkpoint_writes_lightweight_checkpoint(tmp_path: Path) -> None:
    input_path = tmp_path / "model_best.pt"
    output_path = tmp_path / "model_inference.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.tensor([1.0])},
            "model_config": {"vocab_size": 8},
            "optimizer_state_dict": {"state": {}},
            "scheduler_state_dict": {"last_epoch": 1},
        },
        input_path,
    )

    export_inference_checkpoint(input_path, output_path)

    output = torch.load(output_path, map_location="cpu")
    assert output["checkpoint_type"] == "inference"
    assert "optimizer_state_dict" not in output
    assert "scheduler_state_dict" not in output


def test_checkpoint_path_prefers_existing_inference_checkpoint(tmp_path: Path) -> None:
    best_path = tmp_path / "model_best.pt"
    inference_path = tmp_path / "model_inference.pt"
    inference_path.write_bytes(b"checkpoint")
    config = {
        "stage6_train": {"output_best_checkpoint": str(best_path)},
        "stage7_eval": {"inference_checkpoint": str(inference_path)},
    }

    assert _checkpoint_path(config, None) == inference_path


def test_checkpoint_path_falls_back_to_training_checkpoint(tmp_path: Path) -> None:
    best_path = tmp_path / "model_best.pt"
    inference_path = tmp_path / "model_inference.pt"
    config = {
        "stage6_train": {"output_best_checkpoint": str(best_path)},
        "stage7_eval": {"inference_checkpoint": str(inference_path)},
    }

    assert _checkpoint_path(config, None) == best_path
