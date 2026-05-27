from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from datasets import Dataset


@dataclass
class LengthStats:
    min: int
    max: int
    mean: float
    median: float
    p95: int


@dataclass
class SplitAudit:
    split: str
    file_name: str
    rows: int
    null_pairs: int
    empty_pairs: int
    duplicate_pairs: int
    pl_char_lengths: LengthStats
    en_char_lengths: LengthStats
    pl_word_lengths: LengthStats
    en_word_lengths: LengthStats
    samples: list[dict[str, str]]


def _percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = int(round((len(sorted_values) - 1) * q))
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def _compute_stats(values: list[int]) -> LengthStats:
    if not values:
        return LengthStats(min=0, max=0, mean=0.0, median=0.0, p95=0)
    sorted_values = sorted(values)
    return LengthStats(
        min=sorted_values[0],
        max=sorted_values[-1],
        mean=round(sum(sorted_values) / len(sorted_values), 4),
        median=float(median(sorted_values)),
        p95=_percentile(sorted_values, 0.95),
    )


def _pair_hash(pl_text: str, en_text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(pl_text.encode("utf-8", errors="ignore"))
    h.update(b"\x1f")
    h.update(en_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _extract_samples(dataset: Dataset, sample_size: int) -> list[dict[str, str]]:
    sample_rows: list[dict[str, str]] = []
    if len(dataset) == 0:
        return sample_rows

    step = max(1, len(dataset) // max(1, sample_size))
    for i in range(0, len(dataset), step):
        row = dataset[i]
        translation = row.get("translation", {})
        pl_text = (translation.get("pl") or "").strip()
        en_text = (translation.get("en") or "").strip()
        if pl_text and en_text:
            sample_rows.append({"pl": pl_text, "en": en_text})
        if len(sample_rows) >= sample_size:
            break
    return sample_rows


def audit_split(file_path: Path, split_name: str, sample_size: int = 5) -> SplitAudit:
    dataset = Dataset.from_parquet(str(file_path))

    rows = len(dataset)
    null_pairs = 0
    empty_pairs = 0
    duplicate_pairs = 0

    pl_char_lengths: list[int] = []
    en_char_lengths: list[int] = []
    pl_word_lengths: list[int] = []
    en_word_lengths: list[int] = []
    seen_hashes: set[str] = set()

    for row in dataset:
        translation = row.get("translation")
        if not isinstance(translation, dict):
            null_pairs += 1
            continue

        pl_raw = translation.get("pl")
        en_raw = translation.get("en")

        if pl_raw is None or en_raw is None:
            null_pairs += 1
            continue

        pl_text = str(pl_raw).strip()
        en_text = str(en_raw).strip()

        if not pl_text or not en_text:
            empty_pairs += 1
            continue

        pair_digest = _pair_hash(pl_text, en_text)
        if pair_digest in seen_hashes:
            duplicate_pairs += 1
        else:
            seen_hashes.add(pair_digest)

        pl_char_lengths.append(len(pl_text))
        en_char_lengths.append(len(en_text))
        pl_word_lengths.append(len(pl_text.split()))
        en_word_lengths.append(len(en_text.split()))

    return SplitAudit(
        split=split_name,
        file_name=file_path.name,
        rows=rows,
        null_pairs=null_pairs,
        empty_pairs=empty_pairs,
        duplicate_pairs=duplicate_pairs,
        pl_char_lengths=_compute_stats(pl_char_lengths),
        en_char_lengths=_compute_stats(en_char_lengths),
        pl_word_lengths=_compute_stats(pl_word_lengths),
        en_word_lengths=_compute_stats(en_word_lengths),
        samples=_extract_samples(dataset, sample_size=sample_size),
    )


def create_manifest(audits: list[SplitAudit], data_dir: Path) -> dict[str, Any]:
    total_rows = sum(a.rows for a in audits)
    total_null_pairs = sum(a.null_pairs for a in audits)
    total_empty_pairs = sum(a.empty_pairs for a in audits)
    total_duplicate_pairs = sum(a.duplicate_pairs for a in audits)

    return {
        "dataset": "Helsinki-NLP/opus-100 en-pl",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "totals": {
            "rows": total_rows,
            "null_pairs": total_null_pairs,
            "empty_pairs": total_empty_pairs,
            "duplicate_pairs": total_duplicate_pairs,
        },
        "splits": [asdict(audit) for audit in audits],
    }


def write_manifest_json(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")


def write_report_markdown(manifest: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    totals = manifest["totals"]
    lines: list[str] = []
    lines.append("# Data Audit Report (OPUS-100 en-pl)")
    lines.append("")
    lines.append(f"Generated at (UTC): `{manifest['generated_at_utc']}`")
    lines.append(f"Data directory: `{manifest['data_dir']}`")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Rows: {totals['rows']}")
    lines.append(f"- Null pairs: {totals['null_pairs']}")
    lines.append(f"- Empty pairs: {totals['empty_pairs']}")
    lines.append(f"- Duplicate pairs: {totals['duplicate_pairs']}")
    lines.append("")

    for split in manifest["splits"]:
        lines.append(f"## Split: {split['split']}")
        lines.append("")
        lines.append(f"- File: `{split['file_name']}`")
        lines.append(f"- Rows: {split['rows']}")
        lines.append(f"- Null pairs: {split['null_pairs']}")
        lines.append(f"- Empty pairs: {split['empty_pairs']}")
        lines.append(f"- Duplicate pairs: {split['duplicate_pairs']}")
        lines.append("")
        lines.append("### Length Stats")
        lines.append("")
        lines.append(
            "- PL chars (min/mean/median/p95/max): "
            f"{split['pl_char_lengths']['min']}/"
            f"{split['pl_char_lengths']['mean']}/"
            f"{split['pl_char_lengths']['median']}/"
            f"{split['pl_char_lengths']['p95']}/"
            f"{split['pl_char_lengths']['max']}"
        )
        lines.append(
            "- EN chars (min/mean/median/p95/max): "
            f"{split['en_char_lengths']['min']}/"
            f"{split['en_char_lengths']['mean']}/"
            f"{split['en_char_lengths']['median']}/"
            f"{split['en_char_lengths']['p95']}/"
            f"{split['en_char_lengths']['max']}"
        )
        lines.append(
            "- PL words (min/mean/median/p95/max): "
            f"{split['pl_word_lengths']['min']}/"
            f"{split['pl_word_lengths']['mean']}/"
            f"{split['pl_word_lengths']['median']}/"
            f"{split['pl_word_lengths']['p95']}/"
            f"{split['pl_word_lengths']['max']}"
        )
        lines.append(
            "- EN words (min/mean/median/p95/max): "
            f"{split['en_word_lengths']['min']}/"
            f"{split['en_word_lengths']['mean']}/"
            f"{split['en_word_lengths']['median']}/"
            f"{split['en_word_lengths']['p95']}/"
            f"{split['en_word_lengths']['max']}"
        )
        lines.append("")
        lines.append("### Samples")
        lines.append("")

        samples = split.get("samples", [])
        if not samples:
            lines.append("- No valid non-empty samples found.")
            lines.append("")
            continue

        for idx, sample in enumerate(samples, start=1):
            lines.append(f"{idx}. PL: {sample['pl']}")
            lines.append(f"   EN: {sample['en']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
