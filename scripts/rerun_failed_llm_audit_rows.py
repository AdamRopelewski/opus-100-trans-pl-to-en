from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_nested, load_config
from src.utils.llm_audit import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    LlmAuditConfig,
    _build_batches,
    _ollama_generate,
    _prompt_for_batch,
    _validate_response,
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
from src.utils.preaudit import PreAuditConfig, preaudit_filter_rows


SPLITS = ("validation", "test", "train")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rerun rows from LLM audit batches where attempt=2 failed, then write merged audit files."
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/llm_audit_20260601_073335"),
        help="Audit report directory with original llm_audit_batches.jsonl and llm_audit_labels.jsonl.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/project_config.yaml"),
        help="Project config YAML.",
    )
    parser.add_argument("--suffix", default="attempt2_rerun_merged", help="Suffix for merged output files.")
    parser.add_argument("--merge-only", action="store_true", help="Skip Ollama; only merge existing rerun checkpoint files.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for failed original batches. For smoke tests only.",
    )
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def _resolve_split_file(data_dir: Path, split_name: str, pattern: str) -> Path:
    matches = sorted(data_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one parquet for split '{split_name}' in '{data_dir}' using '{pattern}', found {len(matches)}."
        )
    return matches[0]


def _has_failed_second_attempt(batch_log: dict[str, Any]) -> bool:
    return any(
        attempt.get("attempt") == 2 and attempt.get("ok") is False
        for attempt in batch_log.get("attempts", [])
    )


def _label_from_valid(valid: dict[str, list[int]], local_id: int) -> str:
    for label in ("good", "bad", "uncertain"):
        if local_id in valid[label]:
            return label
    return "uncertain"


def _rerun_one_row(cfg: LlmAuditConfig, rec: dict[str, Any]) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    prompt = _prompt_for_batch([rec])
    try:
        raw = _ollama_generate(cfg, prompt)
        valid = _validate_response(raw, {0})
        attempts = [
            {
                "attempt": 1,
                "ok": True,
                "uncertain_count": len(valid["uncertain"]),
                "batch_size": 1,
                "uncertain_ratio": round(len(valid["uncertain"]), 4),
                "rerun_due_to_uncertain": False,
            }
        ]
        return valid, attempts
    except Exception as exc:
        return {"good": [], "bad": [], "uncertain": [0]}, [{"attempt": 1, "ok": False, "error": str(exc)}]


def _parse_label_line(obj: dict[str, Any]) -> tuple[str, int, str]:
    split = str(obj.get("s", obj.get("split")))
    row_index = int(obj.get("i", obj.get("row_index")))
    if "l" in obj:
        label = INT_TO_LABEL[int(obj["l"])]
    else:
        label = str(obj["label"])
    return split, row_index, label


def _load_rerun_labels(path: Path) -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    if not path.exists():
        return labels
    for obj in _load_jsonl(path):
        split, row_index, label = _parse_label_line(obj)
        labels[(split, row_index)] = label
    return labels


def _count_labels_by_split(labels_path: Path) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    with labels_path.open(encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            split, _, _ = _parse_label_line(json.loads(line))
            counts[split] += 1
    return counts


def _count_merged_batches_by_split(
    original_batches: list[dict[str, Any]],
    failed_batch_keys: set[tuple[str, int]],
    failed_rows_by_batch: dict[tuple[str, int], list[dict[str, Any]]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for batch_log in original_batches:
        split = str(batch_log["split"])
        key = (split, int(batch_log["batch_index"]))
        if key in failed_batch_keys:
            counts[split] += len(failed_rows_by_batch[key])
        else:
            counts[split] += 1
    return counts


def _write_merged_labels(
    original_labels_path: Path,
    out_path: Path,
    split_out_paths: dict[str, Path],
    rerun_labels: dict[tuple[str, int], str],
    split_totals: dict[str, int],
) -> None:
    split_fps = {split: path.open("w", encoding="utf-8") for split, path in split_out_paths.items()}
    bars = {
        split: tqdm(total=split_totals[split], desc=f"Merge labels [{split}]", unit="row")
        for split in SPLITS
        if split_totals.get(split, 0) > 0
    }
    try:
        with original_labels_path.open(encoding="utf-8") as in_fp, out_path.open("w", encoding="utf-8") as out_fp:
            for line in in_fp:
                if not line.strip():
                    continue
                obj = json.loads(line)
                split, row_index, old_label = _parse_label_line(obj)
                label = rerun_labels.get((split, row_index), old_label)
                global_obj = {"s": split, "i": row_index, "l": LABEL_TO_INT[label]}
                split_obj = {"i": row_index, "l": LABEL_TO_INT[label]}
                out_fp.write(json.dumps(global_obj, ensure_ascii=True, separators=(",", ":")) + "\n")
                if split in split_fps:
                    split_fps[split].write(json.dumps(split_obj, ensure_ascii=True, separators=(",", ":")) + "\n")
                if split in bars:
                    bars[split].update(1)
    finally:
        for bar in bars.values():
            bar.close()
        for fp in split_fps.values():
            fp.close()


def _write_merged_batches(
    original_batches: list[dict[str, Any]],
    failed_batch_keys: set[tuple[str, int]],
    failed_rows_by_batch: dict[tuple[str, int], list[dict[str, Any]]],
    rerun_batch_logs: dict[tuple[str, int], dict[str, Any]],
    out_path: Path,
    split_out_paths: dict[str, Path],
    split_totals: dict[str, int],
) -> None:
    split_next_index: dict[str, int] = defaultdict(int)
    split_fps = {split: path.open("w", encoding="utf-8") for split, path in split_out_paths.items()}
    bars = {
        split: tqdm(total=split_totals[split], desc=f"Merge batches [{split}]", unit="batch")
        for split in SPLITS
        if split_totals.get(split, 0) > 0
    }
    try:
        with out_path.open("w", encoding="utf-8") as out_fp:
            for batch_log in original_batches:
                split = str(batch_log["split"])
                original_batch_index = int(batch_log["batch_index"])
                key = (split, original_batch_index)
                if key not in failed_batch_keys:
                    merged = dict(batch_log)
                    merged["batch_index"] = split_next_index[split]
                    split_next_index[split] += 1
                    line = json.dumps(merged, ensure_ascii=True, separators=(",", ":")) + "\n"
                    out_fp.write(line)
                    if split in split_fps:
                        split_fps[split].write(line)
                    if split in bars:
                        bars[split].update(1)
                    continue

                for row in failed_rows_by_batch[key]:
                    row_key = (split, int(row["row_index"]))
                    rerun_log = rerun_batch_logs[row_key]
                    merged = dict(rerun_log)
                    merged["batch_index"] = split_next_index[split]
                    split_next_index[split] += 1
                    line = json.dumps(merged, ensure_ascii=True, separators=(",", ":")) + "\n"
                    out_fp.write(line)
                    if split in split_fps:
                        split_fps[split].write(line)
                    if split in bars:
                        bars[split].update(1)
    finally:
        for bar in bars.values():
            bar.close()
        for fp in split_fps.values():
            fp.close()


def main() -> int:
    args = _build_parser().parse_args()
    config = load_config(args.config)
    report_dir = args.report_dir
    original_batches_path = report_dir / "llm_audit_batches.jsonl"
    original_labels_path = report_dir / "llm_audit_labels.jsonl"
    if not original_batches_path.exists() or not original_labels_path.exists():
        raise FileNotFoundError(f"Missing original audit files in {report_dir}")

    data_dir = Path(get_nested(config, "paths.raw_data_dir", "data/raw/en-pl"))
    split_patterns = {
        "validation": str(get_nested(config, "dataset.splits.validation_pattern", "validation-*.parquet")),
        "test": str(get_nested(config, "dataset.splits.test_pattern", "test-*.parquet")),
        "train": str(get_nested(config, "dataset.splits.train_pattern", "train-*.parquet")),
    }
    split_files = {split: _resolve_split_file(data_dir, split, split_patterns[split]) for split in SPLITS}

    llm_cfg = LlmAuditConfig(
        model=str(get_nested(config, "stage1_audit.llm.model", DEFAULT_LLM_MODEL)),
        endpoint=str(get_nested(config, "stage1_audit.llm.endpoint", DEFAULT_OLLAMA_ENDPOINT)),
        batch_max_chars=int(get_nested(config, "stage1_audit.llm.batch_max_chars", DEFAULT_LLM_BATCH_MAX_CHARS)),
        max_rows_per_batch=int(get_nested(config, "stage1_audit.llm.max_rows_per_batch", DEFAULT_LLM_MAX_ROWS_PER_BATCH)),
        temperature=float(get_nested(config, "stage1_audit.llm.temperature", DEFAULT_LLM_TEMPERATURE)),
        max_batch_retries=int(get_nested(config, "stage1_audit.llm.max_batch_retries", DEFAULT_LLM_MAX_BATCH_RETRIES)),
        uncertain_ratio_rerun_threshold=float(
            get_nested(config, "stage1_audit.llm.uncertain_ratio_rerun_threshold", DEFAULT_LLM_UNCERTAIN_RATIO_RERUN_THRESHOLD)
        ),
    )
    preaudit_cfg = PreAuditConfig(
        deduplicate_pairs=bool(get_nested(config, "stage1_audit.preaudit.deduplicate_pairs", True)),
        remove_identical_pairs=bool(get_nested(config, "stage1_audit.preaudit.remove_identical_pairs", True)),
        remove_square_bracket_content=bool(get_nested(config, "stage1_audit.preaudit.remove_square_bracket_content", True)),
        min_words=int(get_nested(config, "stage1_audit.preaudit.min_words", 1)),
        max_words=int(get_nested(config, "stage1_audit.preaudit.max_words", 200)),
        max_length_ratio=float(get_nested(config, "stage1_audit.preaudit.max_length_ratio", 4.0)),
    )

    original_batches = _load_jsonl(original_batches_path)
    failed_batch_keys = {
        (str(batch_log["split"]), int(batch_log["batch_index"]))
        for batch_log in original_batches
        if _has_failed_second_attempt(batch_log)
    }
    if args.limit is not None:
        limited = set(sorted(failed_batch_keys)[: args.limit])
        failed_batch_keys = limited

    failed_ids_path = report_dir / "llm_audit_attempt2_failed_ids.jsonl"
    rerun_labels_path = report_dir / "llm_audit_attempt2_rerun_labels.jsonl"
    rerun_batches_path = report_dir / "llm_audit_attempt2_rerun_batches.jsonl"
    merged_labels_path = report_dir / f"llm_audit_labels_{args.suffix}.jsonl"
    merged_batches_path = report_dir / f"llm_audit_batches_{args.suffix}.jsonl"
    merged_label_split_paths = {split: report_dir / f"llm_audit_labels_{split}_{args.suffix}.jsonl" for split in SPLITS}
    merged_batch_split_paths = {split: report_dir / f"llm_audit_batches_{split}_{args.suffix}.jsonl" for split in SPLITS}

    failed_rows_by_batch: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    failed_id_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        needed_indices = {batch_index for batch_split, batch_index in failed_batch_keys if batch_split == split}
        if not needed_indices:
            continue
        tqdm.write(f"Rebuilding batches for {split}: {len(needed_indices)} failed original batches")
        rows, _ = preaudit_filter_rows(
            Dataset.from_parquet(str(split_files[split])),
            preaudit_cfg,
            split_name=split,
            show_progress=True,
        )
        batches = _build_batches(rows, max_chars=llm_cfg.batch_max_chars, max_rows=llm_cfg.max_rows_per_batch)
        for batch_index in sorted(needed_indices):
            batch = batches[batch_index]
            for local_id, rec in enumerate(batch):
                row = {
                    "split": split,
                    "batch_index": batch_index,
                    "local_id": local_id,
                    "row_index": int(rec["row_index"]),
                }
                failed_id_rows.append(row)
                failed_rows_by_batch[(split, batch_index)].append(rec)
    _write_jsonl(failed_ids_path, failed_id_rows)

    rerun_labels = _load_rerun_labels(rerun_labels_path)
    rerun_batch_logs = (
        {(str(obj["split"]), int(obj["row_index"])): obj for obj in _load_jsonl(rerun_batches_path)}
        if rerun_batches_path.exists()
        else {}
    )

    if not args.merge_only:
        rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in failed_id_rows:
            rows_by_split[str(row["split"])].append(row)

        with rerun_labels_path.open("a", encoding="utf-8") as labels_fp, rerun_batches_path.open("a", encoding="utf-8") as batches_fp:
            done = set(rerun_labels)
            for split in SPLITS:
                split_rows = rows_by_split.get(split, [])
                if not split_rows:
                    continue
                completed = sum(1 for row in split_rows if (split, int(row["row_index"])) in done)
                with tqdm(total=len(split_rows), initial=completed, desc=f"Ollama rerun [{split}]", unit="row") as bar:
                    for row in split_rows:
                        row_index = int(row["row_index"])
                        row_key = (split, row_index)
                        if row_key in done:
                            continue
                        rec = failed_rows_by_batch[(split, int(row["batch_index"]))][int(row["local_id"])]
                        valid, attempts = _rerun_one_row(llm_cfg, rec)
                        label = _label_from_valid(valid, 0)
                        label_obj = {"s": split, "i": row_index, "l": LABEL_TO_INT[label]}
                        batch_obj = {
                            "split": split,
                            "batch_index": None,
                            "size": 1,
                            "attempts": attempts,
                            "uncertain_triggered": False,
                            "result": valid,
                            "row_index": row_index,
                            "source": {
                                "original_batch_index": int(row["batch_index"]),
                                "original_local_id": int(row["local_id"]),
                            },
                        }
                        labels_fp.write(json.dumps(label_obj, ensure_ascii=True, separators=(",", ":")) + "\n")
                        batches_fp.write(json.dumps(batch_obj, ensure_ascii=True, separators=(",", ":")) + "\n")
                        labels_fp.flush()
                        batches_fp.flush()
                        rerun_labels[row_key] = label
                        rerun_batch_logs[row_key] = batch_obj
                        done.add(row_key)
                        bar.update(1)

    missing = [(row["split"], int(row["row_index"])) for row in failed_id_rows if (row["split"], int(row["row_index"])) not in rerun_labels]
    if missing:
        raise RuntimeError(f"Missing rerun labels for {len(missing)} rows. Run without --merge-only to finish rerun.")

    label_split_totals = _count_labels_by_split(original_labels_path)
    batch_split_totals = _count_merged_batches_by_split(original_batches, failed_batch_keys, failed_rows_by_batch)

    _write_merged_labels(original_labels_path, merged_labels_path, merged_label_split_paths, rerun_labels, label_split_totals)
    _write_merged_batches(
        original_batches,
        failed_batch_keys,
        failed_rows_by_batch,
        rerun_batch_logs,
        merged_batches_path,
        merged_batch_split_paths,
        batch_split_totals,
    )

    print(f"Failed IDs: {failed_ids_path}")
    print(f"Rerun labels checkpoint: {rerun_labels_path}")
    print(f"Rerun batches checkpoint: {rerun_batches_path}")
    print(f"Merged labels: {merged_labels_path}")
    print(f"Merged batches: {merged_batches_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
