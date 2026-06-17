from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_model import (
    _batched_beam_search_decode,
    _batched_greedy_decode,
    _checkpoint_path,
    _clear_cuda_cache,
    _chunks,
    _compute_metrics,
    _decode_piece_ids,
    _load_model_bundle,
    _resolve_device,
    _stage7_settings,
    _write_metrics_json,
    _write_samples_markdown,
    _write_translations_tsv,
    tqdm,
)
from src.utils.config import get_nested, load_config


DEFAULT_CONFIG = "configs/tatoeba_config.yaml"
EvalRow = tuple[str, str, list[int]]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate PL->EN Transformer on tokenized Tatoeba test shards."
    )
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--decode", choices=("beam", "greedy"), default="beam")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _tokenized_test_paths(config_data: dict[str, Any]) -> tuple[Path, ...]:
    tokenized_dir = Path(
        get_nested(
            config_data,
            "stage4_dataloader.tokenized_splits_dir",
            "data/processed/tatoeba-en-pl/tokenized",
        )
    )
    pattern = str(
        get_nested(
            config_data,
            "stage4_dataloader.tokenized_test_pattern",
            "test-tokenized-*.parquet",
        )
    )
    matches = tuple(sorted((tokenized_dir / "test").glob(pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No tokenized test parquet files in '{tokenized_dir / 'test'}' using '{pattern}'."
        )
    return matches


def _iter_tokenized_rows(
    paths: tuple[Path, ...], tokenizer: Any, batch_size: int
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        names = parquet_file.schema_arrow.names
        if "src_ids" not in names or "tgt_ids" not in names:
            raise ValueError(f"Expected columns 'src_ids' and 'tgt_ids' in {path}.")
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=["src_ids", "tgt_ids"]
        ):
            data = batch.to_pydict()
            for src_ids, tgt_ids in zip(data["src_ids"], data["tgt_ids"], strict=True):
                source_ids = list(src_ids)
                rows.append(
                    (
                        _decode_piece_ids(tokenizer, source_ids),
                        _decode_piece_ids(tokenizer, list(tgt_ids)),
                        source_ids,
                    )
                )
    return rows


def _source_ids_tensor(
    rows: list[list[int]], pad_id: int, max_source_len: int, device: torch.device
) -> torch.Tensor:
    too_long = [len(tokens) for tokens in rows if len(tokens) > max_source_len]
    if too_long:
        raise ValueError(
            f"Source too long: {max(too_long)} tokens > max_seq_len={max_source_len}"
        )
    max_len = max(len(tokens) for tokens in rows)
    padded = [tokens + [pad_id] * (max_len - len(tokens)) for tokens in rows]
    return torch.tensor(padded, dtype=torch.long, device=device)


def _translate_tokenized_batch(
    source_id_rows: list[list[int]],
    tokenizer: Any,
    model: Any,
    model_cfg: Any,
    device: torch.device,
    decode: str,
    max_new_tokens: int,
    beam_size: int,
    length_penalty: float,
) -> list[str]:
    src_ids = _source_ids_tensor(
        source_id_rows, int(tokenizer.pad_id()), int(model_cfg.max_seq_len), device
    )
    bos_id = int(tokenizer.bos_id())
    eos_id = int(tokenizer.eos_id())
    if decode == "greedy":
        output_ids = _batched_greedy_decode(model, src_ids, bos_id, eos_id, max_new_tokens)
    else:
        output_ids = _batched_beam_search_decode(
            model, src_ids, bos_id, eos_id, max_new_tokens, beam_size, length_penalty
        )
    return [_decode_piece_ids(tokenizer, ids) for ids in output_ids]


def _evaluate_rows(
    rows: list[EvalRow],
    tokenizer: Any,
    model: Any,
    model_cfg: Any,
    device: torch.device,
    decode: str,
    settings: dict[str, Any],
) -> list[tuple[int, str, str, str]]:
    outputs: list[tuple[int, str, str, str]] = []
    max_new_tokens = min(int(settings["max_new_tokens"]), int(model_cfg.max_seq_len) - 1)
    row_batches = _chunks(list(enumerate(rows)), int(settings["batch_size"]))
    iterator = (
        tqdm(row_batches, desc="Evaluating", unit="batch")
        if tqdm is not None
        else row_batches
    )
    for batch in iterator:
        indices = [index for index, _row in batch]
        sources = [source for _index, (source, _reference, _ids) in batch]
        references = [reference for _index, (_source, reference, _ids) in batch]
        source_id_rows = [ids for _index, (_source, _reference, ids) in batch]
        hypotheses = _translate_tokenized_batch(
            source_id_rows,
            tokenizer,
            model,
            model_cfg,
            device,
            decode,
            max_new_tokens,
            int(settings["beam_size"]),
            float(settings["length_penalty"]),
        )
        outputs.extend(
            (index, source, reference, hypothesis)
            for index, source, reference, hypothesis in zip(
                indices, sources, references, hypotheses, strict=True
            )
        )
        _clear_cuda_cache(device)
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
        test_paths = _tokenized_test_paths(config_data)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    settings = _stage7_settings(config_data)
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    device = _resolve_device(args.device)
    test_split_label = ", ".join(str(path) for path in test_paths)
    print(
        f"Evaluating checkpoint {checkpoint_path} on tokenized test split "
        f"{test_split_label} using device {device}."
    )
    tokenizer, model, model_cfg = _load_model_bundle(config_data, checkpoint_path, device)
    max_supported_new_tokens = max(1, int(model_cfg.max_seq_len) - 1)
    if int(settings["max_new_tokens"]) > max_supported_new_tokens:
        print(
            f"Capping max_new_tokens from {settings['max_new_tokens']} "
            f"to {max_supported_new_tokens} for max_seq_len={model_cfg.max_seq_len}."
        )
        settings["max_new_tokens"] = max_supported_new_tokens

    rows = _iter_tokenized_rows(
        test_paths,
        tokenizer,
        int(get_nested(config_data, "stage4_dataloader.tokenized_read_batch_size", 8192)),
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print("No tokenized test rows to evaluate.", file=sys.stderr)
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
            "test_split": test_split_label,
            "rows_evaluated": len(translations),
            "decode": args.decode,
            "beam_size": settings["beam_size"],
            "batch_size": settings["batch_size"],
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
