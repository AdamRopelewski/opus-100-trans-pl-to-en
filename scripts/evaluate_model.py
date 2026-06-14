from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.translate import (
    _checkpoint_path,
    _load_sentencepiece,
    _model_config,
    _resolve_device,
    _tokenizer_path,
    _translate_text,
)
from src.model.transformer_nmt import TransformerNMT
from src.utils.config import get_nested, load_config
from src.utils.tokenizer import iter_parallel_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PL->EN Transformer checkpoint on the test split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/project_config.yaml"),
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Model checkpoint path. Defaults to stage7_eval.inference_checkpoint when it exists, then stage6_train.output_best_checkpoint.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--decode",
        choices=("beam", "greedy"),
        default="beam",
        help="Decoding strategy.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N rows.")
    return parser


def _test_split_path(config_data: dict[str, Any]) -> Path:
    processed_dir = Path(
        get_nested(
            config_data, "stage2_cleaning.outputs.processed_dir", "data/processed/en-pl"
        )
    )
    pattern = str(get_nested(config_data, "dataset.splits.test_pattern", "test-*.parquet"))
    matches = sorted(processed_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one test parquet in '{processed_dir}' using '{pattern}', found {len(matches)}."
        )
    return matches[0]


def _stage7_settings(config_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "beam_size": int(get_nested(config_data, "stage7_eval.beam_size", 4)),
        "length_penalty": float(get_nested(config_data, "stage7_eval.length_penalty", 1.0)),
        "max_new_tokens": int(get_nested(config_data, "stage7_eval.max_new_tokens", 128)),
        "num_sample_translations": int(
            get_nested(config_data, "stage7_eval.num_sample_translations", 100)
        ),
        "metrics_json": Path(
            get_nested(
                config_data,
                "stage7_eval.outputs.metrics_json",
                "reports/eval_metrics.json",
            )
        ),
        "translations_tsv": Path(
            get_nested(
                config_data,
                "stage7_eval.outputs.translations_tsv",
                "reports/translations/test_translations.tsv",
            )
        ),
        "samples_md": Path(
            get_nested(
                config_data,
                "stage7_eval.outputs.samples_md",
                "reports/translations/sample_translations.md",
            )
        ),
    }


def _compute_metrics(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    try:
        import sacrebleu
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install sacrebleu to run Stage 7 eval.") from exc
    return {
        "sacrebleu": float(sacrebleu.corpus_bleu(hypotheses, [references]).score),
        "chrf": float(sacrebleu.corpus_chrf(hypotheses, [references]).score),
    }


def _write_translations_tsv(
    path: Path, rows: list[tuple[int, str, str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["index", "source_pl", "reference_en", "hypothesis_en"])
        writer.writerows(rows)


def _write_samples_markdown(
    path: Path, rows: list[tuple[int, str, str, str]], limit: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Sample Translations", ""]
    for index, source, reference, hypothesis in rows[:limit]:
        lines.extend(
            [
                f"## Row {index}",
                "",
                f"PL: {source}",
                "",
                f"Reference EN: {reference}",
                "",
                f"Hypothesis EN: {hypothesis}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_model_bundle(
    config_data: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> tuple[Any, TransformerNMT, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = _load_sentencepiece(_tokenizer_path(config_data, checkpoint))
    model_cfg = _model_config(
        config_data, checkpoint, int(tokenizer.vocab_size()), int(tokenizer.pad_id())
    )
    model = TransformerNMT(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return tokenizer, model, model_cfg


def _evaluate_rows(
    rows: list[tuple[str, str]],
    tokenizer: Any,
    model: TransformerNMT,
    model_cfg: Any,
    device: torch.device,
    decode: str,
    settings: dict[str, Any],
) -> list[tuple[int, str, str, str]]:
    outputs: list[tuple[int, str, str, str]] = []
    for index, (source, reference) in enumerate(rows):
        hypothesis = _translate_text(
            source,
            tokenizer,
            model,
            model_cfg,
            device,
            decode,
            int(settings["max_new_tokens"]),
            int(settings["beam_size"]),
            float(settings["length_penalty"]),
        )
        outputs.append((index, source, reference, hypothesis))
    return outputs


def main() -> int:
    args = _build_parser().parse_args()
    config_data = load_config(args.config)
    if not bool(get_nested(config_data, "stage7_eval.enabled", True)):
        print("stage7_eval.enabled is false, skipping evaluation.")
        return 0

    checkpoint_path = _checkpoint_path(config_data, args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2

    try:
        test_path = _test_split_path(config_data)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    settings = _stage7_settings(config_data)
    device = _resolve_device(args.device)
    tokenizer, model, model_cfg = _load_model_bundle(config_data, checkpoint_path, device)
    rows = list(iter_parallel_rows(test_path))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("No test rows to evaluate.", file=sys.stderr)
        return 4

    translations = _evaluate_rows(
        rows, tokenizer, model, model_cfg, device, args.decode, settings
    )
    references = [row[2] for row in translations]
    hypotheses = [row[3] for row in translations]
    try:
        metrics = _compute_metrics(hypotheses, references)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 5

    _write_translations_tsv(settings["translations_tsv"], translations)
    _write_samples_markdown(
        settings["samples_md"], translations, int(settings["num_sample_translations"])
    )
    _write_metrics_json(
        settings["metrics_json"],
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint_path),
            "test_split": str(test_path),
            "rows_evaluated": len(translations),
            "decode": args.decode,
            "beam_size": settings["beam_size"],
            "length_penalty": settings["length_penalty"],
            "max_new_tokens": settings["max_new_tokens"],
            "metrics": metrics,
        },
    )

    print(
        f"Evaluation complete: rows={len(translations)} "
        f"sacrebleu={metrics['sacrebleu']:.2f} chrf={metrics['chrf']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
