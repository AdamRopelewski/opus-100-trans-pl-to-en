from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.data_audit import Stage1Checks, audit_split


def _write_raw_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path), compression="zstd")


def test_audit_detects_suspicious_pairs(tmp_path: Path) -> None:
    file_path = tmp_path / "validation-00000-of-00001.parquet"
    _write_raw_parquet(
        file_path,
        [
            {"translation": {"pl": "<b>test</b>", "en": "test"}},
            {"translation": {"pl": "123", "en": "999"}},
            {"translation": {"pl": "!!!", "en": "..."}},
        ],
    )

    checks = Stage1Checks(language_id=False)
    result = audit_split(
        file_path=file_path,
        split_name="validation",
        sample_size=3,
        checks=checks,
        random_seed=42,
        source_lang="pl",
        target_lang="en",
        lid_detect=None,
    )

    assert result.rows == 3
    assert result.suspicious_counts.get("html_xml_tags", 0) >= 1
    assert result.suspicious_counts.get("numeric_mismatch", 0) >= 1
    assert result.suspicious_counts.get("punctuation_only", 0) >= 1
    assert len(result.random_samples) > 0


def test_language_id_audit_skips_short_and_numeric_punct(tmp_path: Path) -> None:
    file_path = tmp_path / "train-00000-of-00001.parquet"
    _write_raw_parquet(
        file_path,
        [
            {"translation": {"pl": "ok to dlugie zdanie", "en": "this is long sentence"}},
            {"translation": {"pl": "1", "en": "2"}},
            {"translation": {"pl": "!!!", "en": "..."}},
            {"translation": {"pl": "krotkie", "en": "short"}},
        ],
    )

    checks = Stage1Checks(language_id=True)

    def fake_detect(_text: str) -> str:
        return "xx"

    result = audit_split(
        file_path=file_path,
        split_name="train",
        sample_size=4,
        checks=checks,
        random_seed=42,
        source_lang="pl",
        target_lang="en",
        lid_detect=fake_detect,
    )

    assert result.language_id["mismatch_count"] == 1
    assert result.language_id["mismatch_row_indices"] == [0]
    assert result.language_id["skipped_count"] == 3
