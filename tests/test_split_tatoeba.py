from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.tatoeba import split as split_tatoeba


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")


def _read_rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def test_split_tatoeba_deduplicates_across_shards_and_writes_report(tmp_path: Path) -> None:
    input_dir = tmp_path / "tatoeba-en-pl"
    output_dir = input_dir / "splits"
    report_path = output_dir / "split_manifest.json"
    _write_parquet(
        input_dir / "00000.train.parquet",
        [
            {"pl": "duplikat", "en": "duplicate", "score": 0.9},
            {"pl": "jeden", "en": "one", "score": 0.9},
            {"pl": "dwa", "en": "two", "score": 0.9},
            {"pl": "trzy", "en": "three", "score": 0.9},
        ],
    )
    _write_parquet(
        input_dir / "00001.train.parquet",
        [
            {"pl": "DUPLIKAT", "en": "DUPLICATE", "score": 0.8},
            {"pl": "cztery", "en": "four", "score": 0.8},
            {"pl": "piec", "en": "five", "score": 0.8},
        ],
    )
    args = Namespace(
        input_dir=input_dir,
        input_pattern="*.train.parquet",
        output_dir=output_dir,
        report_json=None,
        validation_size=2,
        test_size=2,
        seed=7,
        overwrite=False,
    )

    report = split_tatoeba.run(args)

    assert report["rows_in"] == 7
    assert report["rows_after_dedup"] == 6
    assert report["rows_removed"] == 1
    assert report["duplicates_dropped"] == 1
    assert report["splits"]["train"]["rows"] == 2
    assert report["splits"]["validation"]["rows"] == 2
    assert report["splits"]["test"]["rows"] == 2
    assert report["leakage_counts"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert report_path.exists()
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["rows_removed"] == 1

    split_paths = [
        output_dir / "train-00000-of-00001.parquet",
        output_dir / "validation-00000-of-00001.parquet",
        output_dir / "test-00000-of-00001.parquet",
    ]
    all_rows = [row for path in split_paths for row in _read_rows(path)]
    keys = {(row["pl"].casefold(), row["en"].casefold()) for row in all_rows}
    assert len(keys) == 6


def test_split_tatoeba_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    input_dir = tmp_path / "tatoeba-en-pl"
    output_dir = input_dir / "splits"
    _write_parquet(
        input_dir / "00000.train.parquet",
        [
            {"pl": "jeden", "en": "one"},
            {"pl": "dwa", "en": "two"},
            {"pl": "trzy", "en": "three"},
        ],
    )
    _write_parquet(output_dir / "train-00000-of-00001.parquet", [{"pl": "old", "en": "old"}])
    args = Namespace(
        input_dir=input_dir,
        input_pattern="*.train.parquet",
        output_dir=output_dir,
        report_json=None,
        validation_size=1,
        test_size=1,
        seed=42,
        overwrite=False,
    )

    try:
        split_tatoeba.run(args)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("Expected overwrite refusal")
