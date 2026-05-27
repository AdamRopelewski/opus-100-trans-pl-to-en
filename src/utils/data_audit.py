from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable

from datasets import Dataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

HTML_TAG_RE = re.compile(r"<\s*/?\s*[A-Za-z][^>]*>")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
NUMBER_RE = re.compile(r"\d+(?:[\.,]\d+)?")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class LengthStats:
    min: int
    max: int
    mean: float
    median: float
    p95: int


@dataclass
class Stage1Checks:
    null_pairs: bool = True
    empty_pairs: bool = True
    duplicate_pairs: bool = True
    language_id: bool = True
    identical_source_target: bool = True
    html_xml_tags: bool = True
    weird_unicode: bool = True
    control_chars: bool = True
    numeric_mismatch: bool = True
    punctuation_only: bool = True
    very_short_pairs: bool = True
    length_stats: bool = True


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
    suspicious_counts: dict[str, int]
    language_id: dict[str, Any]
    random_samples: list[dict[str, str]]
    suspicious_samples: list[dict[str, Any]]


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def _pair_hash(pl_text: str, en_text: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(pl_text.encode("utf-8", errors="ignore"))
    h.update(b"\x1f")
    h.update(en_text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


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


def _has_weird_unicode(text: str) -> bool:
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in {"\t", "\n", "\r"}:
            return True
        if cat in {"So", "Sk"}:
            return True
    return False


def _is_very_short(pl_text: str, en_text: str) -> bool:
    return len(pl_text.split()) <= 1 or len(en_text.split()) <= 1


def _is_numeric_mismatch(pl_text: str, en_text: str) -> bool:
    pl_numbers = sorted(NUMBER_RE.findall(pl_text))
    en_numbers = sorted(NUMBER_RE.findall(en_text))
    return pl_numbers != en_numbers


def _is_language_id_skippable(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    word_count = len(stripped.split())
    if word_count <= 1:
        return True
    if PUNCT_ONLY_RE.fullmatch(stripped):
        return True
    normalized = NUMBER_RE.sub("", stripped)
    normalized = re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)
    return normalized == ""


def _language_id_mismatch(
    pl_text: str,
    en_text: str,
    lid_detect: Callable[[str], str | None] | None,
    source_lang: str,
    target_lang: str,
) -> tuple[bool, list[str]]:
    if lid_detect is None:
        return False, []

    skipped_reasons: list[str] = []
    if _is_language_id_skippable(pl_text):
        skipped_reasons.append("source_short_or_numeric_or_punct")
    if _is_language_id_skippable(en_text):
        skipped_reasons.append("target_short_or_numeric_or_punct")
    if skipped_reasons:
        return False, skipped_reasons

    pl_lang = lid_detect(pl_text)
    en_lang = lid_detect(en_text)
    mismatch = pl_lang not in {source_lang, "pl"} or en_lang not in {target_lang, "en"}
    return mismatch, []


def _row_suspicious_reasons(
    pl_text: str,
    en_text: str,
    checks: Stage1Checks,
    lid_detect,
    source_lang: str,
    target_lang: str,
) -> tuple[list[str], bool, list[str]]:
    reasons: list[str] = []
    if checks.identical_source_target and pl_text == en_text:
        reasons.append("identical_source_target")
    if checks.html_xml_tags and (HTML_TAG_RE.search(pl_text) or HTML_TAG_RE.search(en_text)):
        reasons.append("html_xml_tags")
    if checks.control_chars and (CONTROL_CHAR_RE.search(pl_text) or CONTROL_CHAR_RE.search(en_text)):
        reasons.append("control_chars")
    if checks.weird_unicode and (_has_weird_unicode(pl_text) or _has_weird_unicode(en_text)):
        reasons.append("weird_unicode")
    if checks.punctuation_only and (PUNCT_ONLY_RE.fullmatch(pl_text or "") or PUNCT_ONLY_RE.fullmatch(en_text or "")):
        reasons.append("punctuation_only")
    if checks.very_short_pairs and _is_very_short(pl_text, en_text):
        reasons.append("very_short_pairs")
    if checks.numeric_mismatch and _is_numeric_mismatch(pl_text, en_text):
        reasons.append("numeric_mismatch")
    lid_mismatch = False
    lid_skipped_reasons: list[str] = []
    if checks.language_id:
        lid_mismatch, lid_skipped_reasons = _language_id_mismatch(
            pl_text=pl_text,
            en_text=en_text,
            lid_detect=lid_detect,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        if lid_mismatch:
            reasons.append("language_id_mismatch")
    return reasons, lid_mismatch, lid_skipped_reasons


def audit_split(
    file_path: Path,
    split_name: str,
    sample_size: int,
    checks: Stage1Checks,
    random_seed: int,
    source_lang: str,
    target_lang: str,
    lid_detect,
    show_progress: bool = True,
) -> SplitAudit:
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
    suspicious_counts: Counter[str] = Counter()
    lid_failed_row_indices: list[int] = []
    lid_skipped_row_indices: list[int] = []
    lid_skipped_reason_counts: Counter[str] = Counter()

    valid_rows: list[dict[str, str]] = []
    suspicious_rows: list[dict[str, Any]] = []

    iterator = enumerate(dataset)
    if show_progress and tqdm is not None:
        iterator = enumerate(
            tqdm(
                dataset,
                total=rows,
                desc=f"Stage1 audit [{split_name}]",
                unit="rows",
                leave=False,
            )
        )

    for row_idx, row in iterator:
        translation = row.get("translation")
        if not isinstance(translation, dict):
            null_pairs += 1
            continue

        pl_raw = translation.get("pl")
        en_raw = translation.get("en")
        if pl_raw is None or en_raw is None:
            null_pairs += 1
            continue

        pl_text = str(pl_raw)
        en_text = str(en_raw)

        if checks.empty_pairs and (not pl_text.strip() or not en_text.strip()):
            empty_pairs += 1
            continue

        pl_norm = normalize_text(pl_text)
        en_norm = normalize_text(en_text)

        if checks.duplicate_pairs:
            digest = _pair_hash(pl_norm, en_norm)
            if digest in seen_hashes:
                duplicate_pairs += 1
            else:
                seen_hashes.add(digest)

        if checks.length_stats:
            pl_char_lengths.append(len(pl_norm))
            en_char_lengths.append(len(en_norm))
            pl_word_lengths.append(len(pl_norm.split()))
            en_word_lengths.append(len(en_norm.split()))

        valid_rows.append({"pl": pl_norm, "en": en_norm})

        reasons, lid_mismatch, lid_skipped_reasons = _row_suspicious_reasons(
            pl_norm,
            en_norm,
            checks,
            lid_detect,
            source_lang,
            target_lang,
        )
        if lid_mismatch:
            lid_failed_row_indices.append(row_idx)
        if lid_skipped_reasons:
            lid_skipped_row_indices.append(row_idx)
            lid_skipped_reason_counts.update(lid_skipped_reasons)
        if reasons:
            for reason in reasons:
                suspicious_counts[reason] += 1
            suspicious_rows.append(
                {
                    "row_index": row_idx,
                    "pl": pl_norm,
                    "en": en_norm,
                    "reasons": reasons,
                }
            )

    split_seed = int(hashlib.md5(split_name.encode("utf-8")).hexdigest()[:8], 16)
    sample_seed = random_seed + (split_seed % 100_000)
    rng = random.Random(sample_seed)
    random_samples = rng.sample(valid_rows, min(sample_size, len(valid_rows))) if valid_rows else []
    suspicious_samples = (
        rng.sample(suspicious_rows, min(sample_size, len(suspicious_rows))) if suspicious_rows else []
    )

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
        suspicious_counts=dict(sorted(suspicious_counts.items())),
        language_id={
            "enabled_in_audit": checks.language_id,
            "mismatch_count": len(lid_failed_row_indices),
            "mismatch_row_indices": lid_failed_row_indices,
            "skipped_count": len(lid_skipped_row_indices),
            "skipped_row_indices": lid_skipped_row_indices,
            "skipped_reason_counts": dict(sorted(lid_skipped_reason_counts.items())),
        },
        random_samples=random_samples,
        suspicious_samples=suspicious_samples,
    )


def create_manifest(
    audits: list[SplitAudit],
    data_dir: Path,
    seed: int,
    checks: Stage1Checks,
    language_id_runtime_reason: str,
) -> dict[str, Any]:
    total_rows = sum(a.rows for a in audits)
    total_null_pairs = sum(a.null_pairs for a in audits)
    total_empty_pairs = sum(a.empty_pairs for a in audits)
    total_duplicate_pairs = sum(a.duplicate_pairs for a in audits)

    merged_suspicious: Counter[str] = Counter()
    total_lid_mismatch = 0
    total_lid_skipped = 0
    for audit in audits:
        merged_suspicious.update(audit.suspicious_counts)
        total_lid_mismatch += int(audit.language_id.get("mismatch_count", 0))
        total_lid_skipped += int(audit.language_id.get("skipped_count", 0))

    return {
        "dataset": "Helsinki-NLP/opus-100 en-pl",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "data_dir": str(data_dir),
        "checks": asdict(checks),
        "language_id_runtime": language_id_runtime_reason,
        "totals": {
            "rows": total_rows,
            "null_pairs": total_null_pairs,
            "empty_pairs": total_empty_pairs,
            "duplicate_pairs": total_duplicate_pairs,
            "language_id_mismatch": total_lid_mismatch,
            "language_id_skipped": total_lid_skipped,
            "suspicious": dict(sorted(merged_suspicious.items())),
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
    lines.append(f"Seed: `{manifest['seed']}`")
    lines.append(f"Language ID runtime: `{manifest['language_id_runtime']}`")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Rows: {totals['rows']}")
    lines.append(f"- Null pairs: {totals['null_pairs']}")
    lines.append(f"- Empty pairs: {totals['empty_pairs']}")
    lines.append(f"- Duplicate pairs: {totals['duplicate_pairs']}")
    lines.append(f"- Language ID mismatch: {totals['language_id_mismatch']}")
    lines.append(f"- Language ID skipped: {totals['language_id_skipped']}")
    lines.append(f"- Suspicious counts: {totals['suspicious']}")
    lines.append("")

    for split in manifest["splits"]:
        lines.append(f"## Split: {split['split']}")
        lines.append("")
        lines.append(f"- File: `{split['file_name']}`")
        lines.append(f"- Rows: {split['rows']}")
        lines.append(f"- Null pairs: {split['null_pairs']}")
        lines.append(f"- Empty pairs: {split['empty_pairs']}")
        lines.append(f"- Duplicate pairs: {split['duplicate_pairs']}")
        lines.append(f"- Suspicious counts: {split['suspicious_counts']}")
        lines.append(f"- Language ID audit: {split['language_id']}")
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

        lines.append("### Random Samples")
        lines.append("")
        for idx, sample in enumerate(split.get("random_samples", []), start=1):
            lines.append(f"{idx}. PL: {sample['pl']}")
            lines.append(f"   EN: {sample['en']}")
        if not split.get("random_samples"):
            lines.append("- No random samples available.")
        lines.append("")

        lines.append("### Suspicious Samples")
        lines.append("")
        for idx, sample in enumerate(split.get("suspicious_samples", []), start=1):
            lines.append(f"{idx}. reasons={sample['reasons']}")
            lines.append(f"   PL: {sample['pl']}")
            lines.append(f"   EN: {sample['en']}")
        if not split.get("suspicious_samples"):
            lines.append("- No suspicious samples found.")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
