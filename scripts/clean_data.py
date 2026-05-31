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
    clean_splits,
    create_cleaning_manifest,
    write_cleaning_manifest_json,
    write_cleaning_report_markdown,
)
from src.utils.pipeline_constants import DEFAULT_MAX_LENGTH_RATIO


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
    paths_cfg = {
        "raw_data_dir": Path(get_nested(config_data, "paths.raw_data_dir", "data/raw/en-pl")),
        "processed_data_dir": Path(get_nested(config_data, "paths.processed_data_dir", "data/processed/en-pl")),
        "reports_dir": Path(get_nested(config_data, "paths.reports_dir", "reports")),
        "tokenizer_dir": Path(get_nested(config_data, "paths.tokenizer_dir", "tokenizers")),
        "checkpoints_dir": Path(get_nested(config_data, "paths.checkpoints_dir", "checkpoints")),
        "logs_dir": Path(get_nested(config_data, "paths.logs_dir", "logs")),
        "translations_dir": Path(get_nested(config_data, "paths.translations_dir", "reports/translations")),
    }
    if not bool(get_nested(config_data, "stage2_cleaning.enabled", True)):
        print("stage2_cleaning.enabled is false, skipping cleaning.")
        return 0

    raw_dir = paths_cfg["raw_data_dir"]
    processed_dir = Path(
        get_nested(
            config_data,
            "stage2_cleaning.outputs.processed_dir",
            str(paths_cfg["processed_data_dir"]),
        )
    )
    report_path = Path(
        get_nested(
            config_data,
            "stage2_cleaning.outputs.report_md",
            str(paths_cfg["reports_dir"] / "cleaning_report.md"),
        )
    )
    manifest_path = Path(
        get_nested(
            config_data,
            "stage2_cleaning.outputs.manifest_json",
            str(paths_cfg["reports_dir"] / "cleaning_manifest.json"),
        )
    )
    removed_examples_path = Path(
        get_nested(
            config_data,
            "stage2_cleaning.outputs.removed_examples_jsonl",
            str(paths_cfg["reports_dir"] / "removed_examples.jsonl"),
        )
    )
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
        unicode_normalization=str(get_nested(config_data, "stage2_cleaning.filters.unicode_normalization", "NFKC")),
        strip_whitespace=bool(get_nested(config_data, "stage2_cleaning.filters.strip_whitespace", True)),
        collapse_whitespace=bool(get_nested(config_data, "stage2_cleaning.filters.collapse_whitespace", True)),
        remove_control_chars=bool(get_nested(config_data, "stage2_cleaning.filters.remove_control_chars", True)),
        min_words=int(get_nested(config_data, "stage2_cleaning.filters.min_words", 1)),
        max_words=int(get_nested(config_data, "stage2_cleaning.filters.max_words", 200)),
        max_length_ratio=float(
            get_nested(config_data, "stage2_cleaning.filters.max_length_ratio", DEFAULT_MAX_LENGTH_RATIO)
        ),
        remove_identical_pairs=bool(get_nested(config_data, "stage2_cleaning.filters.remove_identical_pairs", True)),
        dedup_scope=str(get_nested(config_data, "stage2_cleaning.filters.dedup_scope", "global")),
        remove_train_pairs_present_in_validation_or_test=bool(
            get_nested(
                config_data,
                "stage2_cleaning.filters.remove_train_pairs_present_in_validation_or_test",
                True,
            )
        ),
        preserve_validation_test_priority=bool(
            get_nested(config_data, "stage2_cleaning.filters.preserve_validation_test_priority", True)
        ),
    )
    if config.dedup_scope not in {"split", "global"}:
        print(f"Unsupported dedup_scope: {config.dedup_scope}. Use 'split' or 'global'.", file=sys.stderr)
        return 4

    print("Cleaning splits with leakage-safe dedup...")
    split_stats, audit_meta, primary_reason_totals = clean_splits(
        split_files=split_files,
        output_dir=processed_dir,
        config=config,
        removed_examples_path=removed_examples_path,
        show_progress=True,
    )
    for stats in split_stats:
        print(f"Done '{stats.split}': kept {stats.rows_out}/{stats.rows_in} rows")

    manifest = create_cleaning_manifest(
        stats=split_stats,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        config=config,
        audit_meta=audit_meta,
        primary_reason_totals=primary_reason_totals,
        removed_examples_path=removed_examples_path,
    )
    write_cleaning_manifest_json(manifest, manifest_path)
    write_cleaning_report_markdown(manifest, report_path)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
