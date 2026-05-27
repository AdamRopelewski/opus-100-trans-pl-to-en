from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_nested, load_config
from src.utils.data_audit import Stage1Checks, audit_split, create_manifest, write_manifest_json, write_report_markdown
from src.utils.language_id import build_language_id_runtime

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local OPUS-100 en-pl parquet splits.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/project_config.yaml"),
        help="Path to single project config YAML.",
    )
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

    config = load_config(args.config)
    if not bool(get_nested(config, "stage1_audit.enabled", True)):
        print("stage1_audit.enabled is false, skipping audit.")
        return 0

    data_dir = Path(get_nested(config, "paths.raw_data_dir", "data/raw/en-pl"))
    report_path = Path(get_nested(config, "stage1_audit.outputs.report_md", "reports/data_audit.md"))
    manifest_path = Path(get_nested(config, "stage1_audit.outputs.manifest_json", "reports/data_audit_manifest.json"))
    samples_per_split = int(get_nested(config, "stage1_audit.samples_per_split", 5))
    seed = int(get_nested(config, "project.seed", 42))
    checks = Stage1Checks(**dict(get_nested(config, "stage1_audit.checks", {})))
    checks.language_id = True
    source_lang = str(get_nested(config, "dataset.source_lang", "pl"))
    target_lang = str(get_nested(config, "dataset.target_lang", "en"))
    strict_lid_dependency = bool(
        get_nested(config, "stage1_audit.language_id.strict_dependency", False)
    )

    lid_runtime = build_language_id_runtime(True, strict_dependency=strict_lid_dependency)
    if not lid_runtime.enabled:
        print(f"Language ID audit unavailable at runtime: {lid_runtime.reason}")
    split_patterns = {
        "train": str(get_nested(config, "dataset.splits.train_pattern", "train-*.parquet")),
        "validation": str(get_nested(config, "dataset.splits.validation_pattern", "validation-*.parquet")),
        "test": str(get_nested(config, "dataset.splits.test_pattern", "test-*.parquet")),
    }

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 2

    try:
        split_files = {
            split: _resolve_split_file(data_dir, split, split_patterns[split])
            for split in ("train", "validation", "test")
        }
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    audits = []
    split_items = list(split_files.items())
    stage_iter = split_items
    if tqdm is not None:
        stage_iter = tqdm(split_items, desc="Stage1 audit splits", unit="split")

    for split_name, split_path in stage_iter:
        print(f"Auditing split '{split_name}' from '{split_path}'...")
        audits.append(
            audit_split(
                split_path,
                split_name=split_name,
                sample_size=samples_per_split,
                checks=checks,
                random_seed=seed,
                source_lang=source_lang,
                target_lang=target_lang,
                lid_detect=lid_runtime.detect if lid_runtime.enabled else None,
                show_progress=True,
            )
        )

    manifest = create_manifest(
        audits=audits,
        data_dir=data_dir,
        seed=seed,
        checks=checks,
        language_id_runtime_reason=lid_runtime.reason,
    )
    write_manifest_json(manifest, manifest_path)
    write_report_markdown(manifest, report_path)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
