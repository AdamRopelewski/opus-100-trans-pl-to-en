from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate cleaned Tatoeba shards and materialize train/validation/test splits."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/tatoeba-en-pl"))
    parser.add_argument("--input-pattern", default="*.train.parquet")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/tatoeba-en-pl/splits"))
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.validation_size < 0:
        parser.error("--validation-size must be >= 0")
    if args.test_size < 0:
        parser.error("--test-size must be >= 0")
    return args


def list_input_files(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No input Parquet files found in {input_dir} using {pattern!r}.")
    return files


def dedup_key(row: dict[str, Any]) -> tuple[str, str]:
    try:
        return str(row["pl"]).casefold(), str(row["en"]).casefold()
    except KeyError as exc:
        raise ValueError("Input rows must contain 'pl' and 'en' columns.") from exc


def read_deduplicated_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    input_files: list[dict[str, Any]] = []
    duplicates_dropped = 0
    rows_in = 0

    for path in paths:
        table = pq.read_table(path)
        file_rows = table.to_pylist()
        file_duplicates = 0
        rows_in += len(file_rows)
        for row in file_rows:
            key = dedup_key(row)
            if key in seen:
                duplicates_dropped += 1
                file_duplicates += 1
                continue
            seen.add(key)
            rows.append(row)
        input_files.append(
            {
                "path": str(path),
                "rows_in": len(file_rows),
                "duplicates_dropped": file_duplicates,
                "rows_after_dedup_contribution": len(file_rows) - file_duplicates,
            }
        )

    stats = {
        "rows_in": rows_in,
        "rows_after_dedup": len(rows),
        "duplicates_dropped": duplicates_dropped,
        "input_files": input_files,
    }
    return rows, stats


def split_rows(
    rows: list[dict[str, Any]], validation_size: int, test_size: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    if len(rows) < validation_size + test_size:
        raise ValueError(
            f"Not enough rows after dedup for requested splits: rows={len(rows)}, "
            f"validation_size={validation_size}, test_size={test_size}."
        )
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    test_rows = shuffled[:test_size]
    validation_rows = shuffled[test_size : test_size + validation_size]
    train_rows = shuffled[test_size + validation_size :]
    return {"train": train_rows, "validation": validation_rows, "test": test_rows}


def leakage_counts(splits: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    keys = {split: {dedup_key(row) for row in rows} for split, rows in splits.items()}
    return {
        "train_validation": len(keys["train"] & keys["validation"]),
        "train_test": len(keys["train"] & keys["test"]),
        "validation_test": len(keys["validation"] & keys["test"]),
    }


def ensure_writable(output_dir: Path, report_path: Path, overwrite: bool) -> None:
    targets = [output_dir / filename for filename in SPLIT_FILES.values()] + [report_path]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite without --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def write_split_files(splits: dict[str, list[dict[str, Any]]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Any] = {}
    for split, rows in splits.items():
        path = output_dir / SPLIT_FILES[split]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path, compression="zstd")
        written[split] = {"path": str(path), "rows": len(rows)}
    return written


def build_report(
    args: argparse.Namespace,
    input_stats: dict[str, Any],
    written_splits: dict[str, Any],
    leakage: dict[str, int],
) -> dict[str, Any]:
    return {
        "input_dir": str(args.input_dir),
        "input_pattern": args.input_pattern,
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "requested_split_sizes": {
            "validation": args.validation_size,
            "test": args.test_size,
        },
        "rows_in": input_stats["rows_in"],
        "rows_after_dedup": input_stats["rows_after_dedup"],
        "rows_removed": input_stats["duplicates_dropped"],
        "duplicates_dropped": input_stats["duplicates_dropped"],
        "input_files": input_stats["input_files"],
        "splits": written_splits,
        "leakage_counts": leakage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.report_json or (args.output_dir / "split_manifest.json")
    input_files = list_input_files(args.input_dir, args.input_pattern)
    rows, input_stats = read_deduplicated_rows(input_files)
    splits = split_rows(rows, args.validation_size, args.test_size, args.seed)
    leakage = leakage_counts(splits)
    if any(leakage.values()):
        raise ValueError(f"Split leakage detected: {leakage}")
    ensure_writable(args.output_dir, report_path, args.overwrite)
    written_splits = write_split_files(splits, args.output_dir)
    report = build_report(args, input_stats, written_splits, leakage)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    try:
        report = run(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Rows in: {report['rows_in']}")
    print(f"Rows after dedup: {report['rows_after_dedup']}")
    print(f"Rows removed: {report['rows_removed']}")
    for split, info in report["splits"].items():
        print(f"{split}: {info['rows']} -> {info['path']}")
    print(f"Report: {args.report_json or (args.output_dir / 'split_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
