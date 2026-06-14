from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils.config import get_nested, load_config


TRAINING_ONLY_KEYS = {"optimizer_state_dict", "scheduler_state_dict"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a lightweight inference checkpoint without optimizer state."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/project_config.yaml"),
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Training checkpoint. Defaults to stage6_train.output_best_checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Inference checkpoint. Defaults to stage7_eval.inference_checkpoint.",
    )
    return parser


def _input_checkpoint_path(config_data: dict[str, Any], path: Path | None) -> Path:
    if path is not None:
        return path
    return Path(
        get_nested(
            config_data,
            "stage6_train.output_best_checkpoint",
            "checkpoints/model_best.pt",
        )
    )


def _output_checkpoint_path(config_data: dict[str, Any], path: Path | None) -> Path:
    if path is not None:
        return path
    return Path(
        get_nested(
            config_data,
            "stage7_eval.inference_checkpoint",
            "checkpoints/model_inference.pt",
        )
    )


def strip_training_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in ("model_state_dict", "model_config") if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint missing required key(s): {', '.join(missing)}")

    output = {key: value for key, value in checkpoint.items() if key not in TRAINING_ONLY_KEYS}
    output["checkpoint_type"] = "inference"
    output["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    output["stripped_keys"] = sorted(key for key in TRAINING_ONLY_KEYS if key in checkpoint)
    return output


def export_inference_checkpoint(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary.")
    inference_checkpoint = strip_training_state(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_checkpoint, output_path)


def main() -> int:
    args = _build_parser().parse_args()
    config_data = load_config(args.config)
    input_path = _input_checkpoint_path(config_data, args.input)
    output_path = _output_checkpoint_path(config_data, args.output)
    try:
        export_inference_checkpoint(input_path, output_path)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Exported inference checkpoint: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
