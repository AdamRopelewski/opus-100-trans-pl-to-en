from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

WHITESPACE_RE = re.compile(r"\s+")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


@dataclass
class CleaningConfig:
    unicode_normalization: str = "NFKC"
    strip_whitespace: bool = True
    collapse_whitespace: bool = True
    remove_control_chars: bool = True
    min_words: int = 1
    max_words: int = 200
    max_length_ratio: float = 3.0
    remove_identical_pairs: bool = True
    dedup_scope: str = "global"
    remove_train_pairs_present_in_validation_or_test: bool = True
    preserve_validation_test_priority: bool = True


@dataclass
class SplitCleaningStats:
    split: str
    input_file: str
    output_file: str
    rows_in: int
    rows_out: int
    removed_total: int
    primary_reason_counts: dict[str, int]


REASON_PRIORITY: list[str] = [
    "null_pair",
    "empty_pair",
    "identical_source_target",
    "min_words",
    "max_words",
    "length_ratio",
    "train_pair_present_in_validation_or_test",
    "duplicate",
]

def normalize_text(text: str, config: CleaningConfig) -> str:
    out = unicodedata.normalize(config.unicode_normalization, text)
    if config.remove_control_chars:
        out = CONTROL_CHAR_RE.sub("", out)
    if config.collapse_whitespace:
        out = WHITESPACE_RE.sub(" ", out)
    if config.strip_whitespace:
        out = out.strip()
    return out


def pair_hash(source_text: str, target_text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(source_text.encode("utf-8", errors="ignore"))
    h.update(b"\x1f")
    h.update(target_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _word_count(text: str) -> int:
    return len(text.split())


def _pick_primary_reason(reasons: list[str]) -> str:
    for reason in REASON_PRIORITY:
        if reason in reasons:
            return reason
    return reasons[0]


def _row_reasons(
    split_name: str,
    pl_text: str,
    en_text: str,
    config: CleaningConfig,
    pair_digest: str,
    seen_hashes: set[str],
    val_test_hashes: set[str],
) -> list[str]:
    reasons: list[str] = []

    if not pl_text or not en_text:
        reasons.append("empty_pair")
        return reasons

    if config.remove_identical_pairs and pl_text == en_text:
        reasons.append("identical_source_target")

    pl_words = _word_count(pl_text)
    en_words = _word_count(en_text)

    if pl_words < config.min_words or en_words < config.min_words:
        reasons.append("min_words")
    if pl_words > config.max_words or en_words > config.max_words:
        reasons.append("max_words")

    shorter = max(1, min(pl_words, en_words))
    longer = max(pl_words, en_words)
    if (longer / shorter) > config.max_length_ratio:
        reasons.append("length_ratio")

    if (
        split_name == "train"
        and config.remove_train_pairs_present_in_validation_or_test
        and pair_digest in val_test_hashes
    ):
        reasons.append("train_pair_present_in_validation_or_test")

    if pair_digest in seen_hashes:
        reasons.append("duplicate")

    return reasons


def _iter_rows(dataset: Dataset):
    for idx, row in enumerate(dataset):
        yield idx, row


def build_val_test_hashes(
    split_datasets: dict[str, Dataset],
    config: CleaningConfig,
) -> set[str]:
    hashes: set[str] = set()
    for split_name in ("validation", "test"):
        dataset = split_datasets[split_name]
        for _idx, row in _iter_rows(dataset):
            tr = row.get("translation")
            if not isinstance(tr, dict):
                continue
            pl_raw = tr.get("pl")
            en_raw = tr.get("en")
            if pl_raw is None or en_raw is None:
                continue
            pl_text = normalize_text(str(pl_raw), config)
            en_text = normalize_text(str(en_raw), config)
            if not pl_text or not en_text:
                continue
            hashes.add(pair_hash(pl_text, en_text))
    return hashes


def _write_removed_example(fp, payload: dict[str, Any]) -> None:
    fp.write(json.dumps(payload, ensure_ascii=True) + "\n")


def clean_splits(
    split_files: dict[str, Path],
    output_dir: Path,
    config: CleaningConfig,
    removed_examples_path: Path,
    show_progress: bool = True,
) -> tuple[list[SplitCleaningStats], dict[str, Any], dict[str, int]]:
    datasets_map = {split: Dataset.from_parquet(str(path)) for split, path in split_files.items()}

    val_test_hashes = build_val_test_hashes(datasets_map, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    removed_examples_path.parent.mkdir(parents=True, exist_ok=True)

    if config.dedup_scope == "global":
        if config.preserve_validation_test_priority:
            process_order = ["validation", "test", "train"]
        else:
            process_order = ["train", "validation", "test"]
    else:
        process_order = ["train", "validation", "test"]

    global_seen_hashes: set[str] = set()
    split_stats_map: dict[str, SplitCleaningStats] = {}
    primary_totals: Counter[str] = Counter()

    with removed_examples_path.open("w", encoding="utf-8") as removed_fp:
        split_iter = process_order
        if show_progress and tqdm is not None:
            split_iter = tqdm(process_order, desc="Stage2 cleaning splits", unit="split")

        for split_name in split_iter:
            dataset = datasets_map[split_name]
            input_file = split_files[split_name]
            output_file = output_dir / input_file.name

            seen_hashes = global_seen_hashes if config.dedup_scope == "global" else set()
            primary_reason_counts: Counter[str] = Counter()
            cleaned_pl: list[str] = []
            cleaned_en: list[str] = []

            row_iter = _iter_rows(dataset)
            if show_progress and tqdm is not None:
                row_iter = enumerate(
                    tqdm(
                        dataset,
                        total=len(dataset),
                        desc=f"Stage2 clean [{split_name}]",
                        unit="rows",
                        leave=False,
                    )
                )

            for row_idx, row in row_iter:
                tr = row.get("translation")
                reasons: list[str] = []
                pl_text = ""
                en_text = ""

                if not isinstance(tr, dict):
                    reasons = ["null_pair"]
                else:
                    pl_raw = tr.get("pl")
                    en_raw = tr.get("en")
                    if pl_raw is None or en_raw is None:
                        reasons = ["null_pair"]
                    else:
                        pl_text = normalize_text(str(pl_raw), config)
                        en_text = normalize_text(str(en_raw), config)
                        digest = pair_hash(pl_text, en_text)
                        reasons = _row_reasons(
                            split_name=split_name,
                            pl_text=pl_text,
                            en_text=en_text,
                            config=config,
                            pair_digest=digest,
                            seen_hashes=seen_hashes,
                            val_test_hashes=val_test_hashes,
                        )

                if reasons:
                    primary = _pick_primary_reason(reasons)
                    primary_reason_counts[primary] += 1
                    primary_totals[primary] += 1
                    _write_removed_example(
                        removed_fp,
                        {
                            "split": split_name,
                            "row_index": row_idx,
                            "pl": pl_text,
                            "en": en_text,
                            "reason": primary,
                            "all_reasons": reasons,
                        },
                    )
                    continue

                digest = pair_hash(pl_text, en_text)
                seen_hashes.add(digest)
                cleaned_pl.append(pl_text)
                cleaned_en.append(en_text)

            table = pa.table({"pl": cleaned_pl, "en": cleaned_en})
            pq.write_table(table, str(output_file), compression="zstd")

            rows_in = len(dataset)
            rows_out = len(cleaned_pl)
            split_stats_map[split_name] = SplitCleaningStats(
                split=split_name,
                input_file=input_file.name,
                output_file=output_file.name,
                rows_in=rows_in,
                rows_out=rows_out,
                removed_total=rows_in - rows_out,
                primary_reason_counts=dict(sorted(primary_reason_counts.items())),
            )

    ordered_stats = [split_stats_map[split] for split in ("train", "validation", "test")]
    return ordered_stats, {}, dict(sorted(primary_totals.items()))


def create_cleaning_manifest(
    stats: list[SplitCleaningStats],
    raw_dir: Path,
    processed_dir: Path,
    config: CleaningConfig,
    audit_meta: dict[str, Any],
    primary_reason_totals: dict[str, int],
    removed_examples_path: Path,
) -> dict[str, Any]:
    rows_in_total = sum(s.rows_in for s in stats)
    rows_out_total = sum(s.rows_out for s in stats)
    total_removed = rows_in_total - rows_out_total

    return {
        "dataset": "Helsinki-NLP/opus-100 en-pl",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "removed_examples_jsonl": str(removed_examples_path),
        "audit_meta": audit_meta,
        "config": asdict(config),
        "totals": {
            "rows_in": rows_in_total,
            "rows_out": rows_out_total,
            "rows_removed": total_removed,
            "retention_ratio": round(rows_out_total / max(1, rows_in_total), 6),
            "primary_reason_counts": primary_reason_totals,
        },
        "splits": [asdict(s) for s in stats],
    }


def write_cleaning_manifest_json(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def write_cleaning_report_markdown(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = manifest["config"]
    totals = manifest["totals"]
    lines: list[str] = []
    lines.append("# Data Cleaning Report (OPUS-100 en-pl)")
    lines.append("")
    lines.append(f"Generated at (UTC): `{manifest['generated_at_utc']}`")
    lines.append(f"Raw directory: `{manifest['raw_dir']}`")
    lines.append(f"Processed directory: `{manifest['processed_dir']}`")
    lines.append(f"Removed examples JSONL: `{manifest['removed_examples_jsonl']}`")
    lines.append(f"Audit metadata: `{manifest['audit_meta']}`")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- dedup_scope: {cfg['dedup_scope']}")
    lines.append(
        "- preserve_validation_test_priority: "
        f"{cfg['preserve_validation_test_priority']}"
    )
    lines.append(
        "- remove_train_pairs_present_in_validation_or_test: "
        f"{cfg['remove_train_pairs_present_in_validation_or_test']}"
    )
    lines.append(f"- min_words: {cfg['min_words']}")
    lines.append(f"- max_words: {cfg['max_words']}")
    lines.append(f"- max_length_ratio: {cfg['max_length_ratio']}")
    lines.append(f"- unicode_normalization: {cfg['unicode_normalization']}")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Rows in: {totals['rows_in']}")
    lines.append(f"- Rows out: {totals['rows_out']}")
    lines.append(f"- Rows removed: {totals['rows_removed']}")
    lines.append(f"- Retention ratio: {totals['retention_ratio']}")
    lines.append(f"- Primary reason counts: {totals['primary_reason_counts']}")
    lines.append("")

    for split in manifest["splits"]:
        lines.append(f"## Split: {split['split']}")
        lines.append("")
        lines.append(f"- Input file: `{split['input_file']}`")
        lines.append(f"- Output file: `{split['output_file']}`")
        lines.append(f"- Rows in: {split['rows_in']}")
        lines.append(f"- Rows out: {split['rows_out']}")
        lines.append(f"- Rows removed: {split['removed_total']}")
        lines.append(f"- Primary reason counts: {split['primary_reason_counts']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
