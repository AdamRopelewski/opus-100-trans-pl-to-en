from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_nested, load_config
from src.utils.llm_audit import (
    LlmAuditConfig,
    PreAuditConfig,
    UncertainBatchError,
    run_stage1_llm_audit,
    write_llm_audit_report,
)
from src.utils.pipeline_constants import (
    DEFAULT_LLM_BATCH_MAX_CHARS,
    DEFAULT_LLM_MAX_BATCH_RETRIES,
    DEFAULT_LLM_MAX_ROWS_PER_BATCH,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_UNCERTAIN_RATIO_RERUN_THRESHOLD,
    DEFAULT_OLLAMA_ENDPOINT,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local OPUS-100 en-pl parquet splits with local Ollama LLM.")
    parser.add_argument("--config", type=Path, default=Path("configs/project_config.yaml"), help="Path to project config YAML.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode: print Ollama responses and batch preview rows.")
    return parser


def _resolve_split_file(data_dir: Path, split_name: str, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one parquet for split '{split_name}' in '{data_dir}' using '{pattern}', found {len(matches)}."
        )
    return matches[0]


def main() -> int:
    args = _build_parser().parse_args()
    config = load_config(args.config)

    if not bool(get_nested(config, "stage1_audit.enabled", True)):
        print("stage1_audit.enabled is false, skipping audit.")
        return 0

    data_dir = Path(get_nested(config, "paths.raw_data_dir", "data/raw/en-pl"))
    report_path = Path(get_nested(config, "stage1_audit.outputs.report_md", "reports/data_audit.md"))
    manifest_path = Path(get_nested(config, "stage1_audit.outputs.manifest_json", "reports/data_audit_manifest.json"))

    split_patterns = {
        "validation": str(get_nested(config, "dataset.splits.validation_pattern", "validation-*.parquet")),
        "test": str(get_nested(config, "dataset.splits.test_pattern", "test-*.parquet")),
        "train": str(get_nested(config, "dataset.splits.train_pattern", "train-*.parquet")),
    }

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        return 2

    try:
        split_files = {
            split: _resolve_split_file(data_dir, split, split_patterns[split])
            for split in ("validation", "test", "train")
        }
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    llm_cfg = LlmAuditConfig(
        model=str(get_nested(config, "stage1_audit.llm.model", DEFAULT_LLM_MODEL)),
        endpoint=str(get_nested(config, "stage1_audit.llm.endpoint", DEFAULT_OLLAMA_ENDPOINT)),
        batch_max_chars=int(get_nested(config, "stage1_audit.llm.batch_max_chars", DEFAULT_LLM_BATCH_MAX_CHARS)),
        max_rows_per_batch=int(get_nested(config, "stage1_audit.llm.max_rows_per_batch", DEFAULT_LLM_MAX_ROWS_PER_BATCH)),
        temperature=float(get_nested(config, "stage1_audit.llm.temperature", DEFAULT_LLM_TEMPERATURE)),
        max_batch_retries=int(get_nested(config, "stage1_audit.llm.max_batch_retries", DEFAULT_LLM_MAX_BATCH_RETRIES)),
        uncertain_ratio_rerun_threshold=float(
            get_nested(
                config,
                "stage1_audit.llm.uncertain_ratio_rerun_threshold",
                DEFAULT_LLM_UNCERTAIN_RATIO_RERUN_THRESHOLD,
            )
        ),
        verbose=bool(args.verbose or get_nested(config, "stage1_audit.llm.verbose", False)),
        verbose_preview_rows=int(get_nested(config, "stage1_audit.llm.verbose_preview_rows", 10)),
    )

    reports_dir = Path(get_nested(config, "paths.reports_dir", "reports"))
    preaudit_cfg = PreAuditConfig(
        deduplicate_pairs=bool(get_nested(config, "stage1_audit.preaudit.deduplicate_pairs", True)),
        remove_identical_pairs=bool(get_nested(config, "stage1_audit.preaudit.remove_identical_pairs", True)),
        remove_square_bracket_content=bool(
            get_nested(config, "stage1_audit.preaudit.remove_square_bracket_content", True)
        ),
        min_words=int(get_nested(config, "stage1_audit.preaudit.min_words", 1)),
        max_words=int(get_nested(config, "stage1_audit.preaudit.max_words", 200)),
        max_length_ratio=float(get_nested(config, "stage1_audit.preaudit.max_length_ratio", 4.0)),
    )
    while True:
        try:
            manifest = run_stage1_llm_audit(
                split_files=split_files,
                cfg=llm_cfg,
                reports_dir=reports_dir,
                preaudit_cfg=preaudit_cfg,
                show_progress=True,
            )
            break
        except UncertainBatchError as exc:
            print(str(exc), file=sys.stderr)
            answer = input("High uncertain batch detected. Increase retries and continue? [y/N]: ").strip().lower()
            if answer != "y":
                return 5
            llm_cfg.max_batch_retries += 1
            print(f"Retrying full audit with max_batch_retries={llm_cfg.max_batch_retries}...")

    write_llm_audit_report(manifest, out_md=report_path, out_json=manifest_path)

    print(f"Wrote report: {report_path}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote labels: {manifest['artifacts']['labels_jsonl']}")
    print(f"Wrote bad sentences: {manifest['artifacts']['bad_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
