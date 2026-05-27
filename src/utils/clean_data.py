from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class CleaningConfig:
    min_words: int = 1
    max_words: int = 200
    max_length_ratio: float = 3.0
    unicode_normalization: str = "NFKC"
    dedup_scope: str = "split"  # split | global


@dataclass
class SplitCleaningStats:
    split: str
    input_file: str
    output_file: str
    rows_in: int
    rows_out: int
    removed_null_pair: int
    removed_empty_pair: int
    removed_min_words: int
    removed_max_words: int
    removed_length_ratio: int
    removed_duplicates: int


def normalize_text(text: str, normalization: str = "NFKC") -> str:
    normalized = unicodedata.normalize(normalization, text)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def pair_hash(pl_text: str, en_text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(pl_text.encode("utf-8", errors="ignore"))
    h.update(b"\x1f")
    h.update(en_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _word_count(text: str) -> int:
    return len(text.split())


def clean_split(
    input_file: Path,
    output_file: Path,
    split_name: str,
    config: CleaningConfig,
    global_seen_hashes: set[str] | None = None,
) -> SplitCleaningStats:
    dataset = Dataset.from_parquet(str(input_file))

    removed_null_pair = 0
    removed_empty_pair = 0
    removed_min_words = 0
    removed_max_words = 0
    removed_length_ratio = 0
    removed_duplicates = 0

    seen_hashes: set[str] = global_seen_hashes if global_seen_hashes is not None else set()
    cleaned_pl: list[str] = []
    cleaned_en: list[str] = []

    for row in dataset:
        translation = row.get("translation")
        if not isinstance(translation, dict):
            removed_null_pair += 1
            continue

        pl_raw = translation.get("pl")
        en_raw = translation.get("en")
        if pl_raw is None or en_raw is None:
            removed_null_pair += 1
            continue

        pl_text = normalize_text(str(pl_raw), normalization=config.unicode_normalization)
        en_text = normalize_text(str(en_raw), normalization=config.unicode_normalization)

        if not pl_text or not en_text:
            removed_empty_pair += 1
            continue

        pl_words = _word_count(pl_text)
        en_words = _word_count(en_text)

        if pl_words < config.min_words or en_words < config.min_words:
            removed_min_words += 1
            continue

        if pl_words > config.max_words or en_words > config.max_words:
            removed_max_words += 1
            continue

        shorter = max(1, min(pl_words, en_words))
        longer = max(pl_words, en_words)
        ratio = longer / shorter
        if ratio > config.max_length_ratio:
            removed_length_ratio += 1
            continue

        digest = pair_hash(pl_text, en_text)
        if digest in seen_hashes:
            removed_duplicates += 1
            continue

        seen_hashes.add(digest)
        cleaned_pl.append(pl_text)
        cleaned_en.append(en_text)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"pl": cleaned_pl, "en": cleaned_en})
    pq.write_table(table, str(output_file), compression="zstd")

    return SplitCleaningStats(
        split=split_name,
        input_file=input_file.name,
        output_file=output_file.name,
        rows_in=len(dataset),
        rows_out=len(cleaned_pl),
        removed_null_pair=removed_null_pair,
        removed_empty_pair=removed_empty_pair,
        removed_min_words=removed_min_words,
        removed_max_words=removed_max_words,
        removed_length_ratio=removed_length_ratio,
        removed_duplicates=removed_duplicates,
    )


def create_cleaning_manifest(
    stats: list[SplitCleaningStats],
    raw_dir: Path,
    processed_dir: Path,
    config: CleaningConfig,
) -> dict[str, Any]:
    rows_in_total = sum(s.rows_in for s in stats)
    rows_out_total = sum(s.rows_out for s in stats)
    total_removed = rows_in_total - rows_out_total

    return {
        "dataset": "Helsinki-NLP/opus-100 en-pl",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "config": asdict(config),
        "totals": {
            "rows_in": rows_in_total,
            "rows_out": rows_out_total,
            "rows_removed": total_removed,
            "retention_ratio": round(rows_out_total / max(1, rows_in_total), 6),
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
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"- min_words: {cfg['min_words']}")
    lines.append(f"- max_words: {cfg['max_words']}")
    lines.append(f"- max_length_ratio: {cfg['max_length_ratio']}")
    lines.append(f"- unicode_normalization: {cfg['unicode_normalization']}")
    lines.append(f"- dedup_scope: {cfg['dedup_scope']}")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Rows in: {totals['rows_in']}")
    lines.append(f"- Rows out: {totals['rows_out']}")
    lines.append(f"- Rows removed: {totals['rows_removed']}")
    lines.append(f"- Retention ratio: {totals['retention_ratio']}")
    lines.append("")

    for split in manifest["splits"]:
        lines.append(f"## Split: {split['split']}")
        lines.append("")
        lines.append(f"- Input file: `{split['input_file']}`")
        lines.append(f"- Output file: `{split['output_file']}`")
        lines.append(f"- Rows in: {split['rows_in']}")
        lines.append(f"- Rows out: {split['rows_out']}")
        lines.append(f"- Removed null pairs: {split['removed_null_pair']}")
        lines.append(f"- Removed empty pairs: {split['removed_empty_pair']}")
        lines.append(f"- Removed min words: {split['removed_min_words']}")
        lines.append(f"- Removed max words: {split['removed_max_words']}")
        lines.append(f"- Removed length ratio: {split['removed_length_ratio']}")
        lines.append(f"- Removed duplicates: {split['removed_duplicates']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
