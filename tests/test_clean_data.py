from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.clean_data import CleaningConfig, clean_splits, normalize_text


def _write_raw_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path), compression="zstd")


def test_normalize_whitespace() -> None:
    cfg = CleaningConfig()
    text = "  Ala\t\nma   kota  "
    assert normalize_text(text, cfg) == "Ala ma kota"


def test_remove_control_chars() -> None:
    cfg = CleaningConfig(remove_control_chars=True)
    text = "abc\x00\x07def"
    assert normalize_text(text, cfg) == "abcdef"


def test_global_dedup_no_leakage(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    removed_path = tmp_path / "removed.jsonl"

    _write_raw_parquet(
        raw_dir / "train-00000-of-00001.parquet",
        [
            {"translation": {"pl": "wspolna para", "en": "shared pair"}},
            {"translation": {"pl": "tylko train", "en": "train only"}},
        ],
    )
    _write_raw_parquet(
        raw_dir / "validation-00000-of-00001.parquet",
        [{"translation": {"pl": "wspolna para", "en": "shared pair"}}],
    )
    _write_raw_parquet(
        raw_dir / "test-00000-of-00001.parquet",
        [{"translation": {"pl": "tylko test", "en": "test only"}}],
    )

    cfg = CleaningConfig(
        dedup_scope="global",
        remove_train_pairs_present_in_validation_or_test=True,
        preserve_validation_test_priority=True,
    )

    split_files = {
        "train": raw_dir / "train-00000-of-00001.parquet",
        "validation": raw_dir / "validation-00000-of-00001.parquet",
        "test": raw_dir / "test-00000-of-00001.parquet",
    }
    stats, audit_meta, _reasons = clean_splits(split_files, proc_dir, cfg, removed_path)
    by_split = {s.split: s for s in stats}

    assert by_split["validation"].rows_out == 1
    assert by_split["test"].rows_out == 1
    assert by_split["train"].rows_out == 1
    assert by_split["train"].primary_reason_counts["train_pair_present_in_validation_or_test"] == 1
    assert audit_meta == {}


def test_validation_priority_when_duplicate_across_val_test(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    removed_path = tmp_path / "removed.jsonl"

    _write_raw_parquet(
        raw_dir / "train-00000-of-00001.parquet",
        [{"translation": {"pl": "train", "en": "train"}}],
    )
    _write_raw_parquet(
        raw_dir / "validation-00000-of-00001.parquet",
        [{"translation": {"pl": "dupe", "en": "pair"}}],
    )
    _write_raw_parquet(
        raw_dir / "test-00000-of-00001.parquet",
        [{"translation": {"pl": "dupe", "en": "pair"}}],
    )

    cfg = CleaningConfig(
        dedup_scope="global",
        remove_train_pairs_present_in_validation_or_test=False,
        preserve_validation_test_priority=True,
    )

    split_files = {
        "train": raw_dir / "train-00000-of-00001.parquet",
        "validation": raw_dir / "validation-00000-of-00001.parquet",
        "test": raw_dir / "test-00000-of-00001.parquet",
    }
    stats, _audit_meta, _reasons = clean_splits(split_files, proc_dir, cfg, removed_path)
    by_split = {s.split: s for s in stats}

    assert by_split["validation"].rows_out == 1
    assert by_split["test"].rows_out == 0
    assert by_split["test"].primary_reason_counts["duplicate"] == 1
