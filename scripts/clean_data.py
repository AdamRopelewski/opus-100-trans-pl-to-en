from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_nested, load_config
from src.utils.clean_data import (
    CleaningConfig,
    clean_split,
    create_cleaning_manifest,
    write_cleaning_manifest_json,
    write_cleaning_report_markdown,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean local OPUS-100 en-pl parquet splits.")
    parser.add_argument("--config", type=Path, default=Path("configs/project_config.yaml"), help="Path to single project config YAML.")
    return parser


def _resolve_split_file(data_dir: Path, split_name: str, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one parquet for split '{split_name}' in '{data_dir}' using '{pattern}', found {len(matches)}."
        )
    return matches[0]


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    config_data = load_config(args.config)

    raw_dir = Path(get_nested(config_data, "paths.raw_data_dir", "data/raw/en-pl"))
    processed_dir = Path(get_nested(config_data, "stage2_cleaning.outputs.processed_dir", "data/processed/en-pl"))
    report_path = Path(get_nested(config_data, "stage2_cleaning.outputs.report_md", "reports/cleaning_report.md"))
    manifest_path = Path(get_nested(config_data, "stage2_cleaning.outputs.manifest_json", "reports/cleaning_manifest.json"))
    split_patterns = {
        "train": str(get_nested(config_data, "dataset.splits.train_pattern", "train-*.parquet")),
        "validation": str(get_nested(config_data, "dataset.splits.validation_pattern", "validation-*.parquet")),
        "test": str(get_nested(config_data, "dataset.splits.test_pattern", "test-*.parquet")),
    }

    if not raw_dir.exists() or not raw_dir.is_dir():
        print(f"Raw directory not found: {raw_dir}", file=sys.stderr)
        return 2

    try:
        split_files = {
            split: _resolve_split_file(raw_dir, split, split_patterns[split])
            for split in ("train", "validation", "test")
        }
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    config = CleaningConfig(
        min_words=int(get_nested(config_data, "stage2_cleaning.filters.min_words", 1)),
        max_words=int(get_nested(config_data, "stage2_cleaning.filters.max_words", 200)),
        max_length_ratio=float(get_nested(config_data, "stage2_cleaning.filters.max_length_ratio", 3.0)),
        unicode_normalization=str(get_nested(config_data, "stage2_cleaning.filters.unicode_normalization", "NFKC")),
        dedup_scope=str(get_nested(config_data, "stage2_cleaning.filters.dedup_scope", "split")),
    )

    global_seen_hashes: set[str] | None = set() if config.dedup_scope == "global" else None
    split_stats = []

    for split_name in ("train", "validation", "test"):
        input_file = split_files[split_name]
        output_file = processed_dir / input_file.name
        seen_scope = global_seen_hashes if config.dedup_scope == "global" else set()

        print(f"Cleaning split '{split_name}' from '{input_file}'...")
        stats = clean_split(
            input_file=input_file,
            output_file=output_file,
            split_name=split_name,
            config=config,
            global_seen_hashes=seen_scope,
        )
        split_stats.append(stats)
        print(f"Done '{split_name}': kept {stats.rows_out}/{stats.rows_in} rows")

    manifest = create_cleaning_manifest(
        stats=split_stats,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        config=config,
    )
    write_cleaning_manifest_json(manifest, manifest_path)
    write_cleaning_report_markdown(manifest, report_path)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
