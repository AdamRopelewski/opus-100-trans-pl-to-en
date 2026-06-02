from __future__ import annotations

from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.tokenizer import (
    TokenizerConfig,
    collect_long_word_split_stats,
    collect_split_stats,
    common_long_word_examples,
    iter_parallel_rows,
    write_tokenizer_stats_markdown,
    write_training_corpus,
)


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, str(path), compression="zstd")


def test_iter_parallel_rows_supports_cleaned_columns(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    _write_parquet(path, [{"pl": "Zażółć", "en": "Yellow"}])

    assert list(iter_parallel_rows(path)) == [("Zażółć", "Yellow")]


def test_iter_parallel_rows_supports_translation_struct(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    _write_parquet(path, [{"translation": {"pl": "kot", "en": "cat"}}])

    assert list(iter_parallel_rows(path)) == [("kot", "cat")]


def test_write_training_corpus_writes_shared_pl_en_lines(tmp_path: Path) -> None:
    path = tmp_path / "train.parquet"
    output = tmp_path / "spm_train.txt"
    _write_parquet(path, [{"pl": "jeden", "en": "one"}, {"pl": "dwa", "en": "two"}])

    line_count = write_training_corpus(path, output)

    assert line_count == 4
    assert output.read_text(encoding="utf-8").splitlines() == ["jeden", "one", "dwa", "two"]


def test_collect_split_stats_counts_unknowns_and_compression(tmp_path: Path) -> None:
    path = tmp_path / "validation.parquet"
    _write_parquet(path, [{"pl": "aa bb", "en": "cc"}])

    def encode(text: str) -> list[int]:
        return [1 if token == "bb" else 7 for token in text.split()]

    stats = collect_split_stats("validation", path, encode, unk_id=1)

    assert stats.rows == 1
    assert stats.whitespace_words == 3
    assert stats.subword_pieces == 3
    assert stats.unk_pieces == 1
    assert stats.compression_ratio == 1.0
    assert stats.unk_rate == 1 / 3


def test_long_word_stats_show_common_words_split_less() -> None:
    frequencies = Counter({"commonlongword": 100, "rarelongwordx": 1})

    def encode(text: str) -> list[int]:
        if text == "commonlongword":
            return [10]
        return [10, 11, 12]

    stats = collect_long_word_split_stats(frequencies, encode, min_length=12)
    examples = common_long_word_examples(frequencies, encode, min_length=12)

    assert stats["freq_gt_50"]["mean_pieces_per_word"] == 1.0
    assert stats["freq_1"]["mean_pieces_per_word"] == 3.0
    assert examples == [("commonlongword", 100, 1), ("rarelongwordx", 1, 3)]


def test_write_tokenizer_stats_markdown(tmp_path: Path) -> None:
    report = tmp_path / "tokenizer_stats.md"
    write_tokenizer_stats_markdown(
        report,
        TokenizerConfig(vocab_size=16000),
        corpus_lines=10,
        split_stats=[],
        long_word_stats={"freq_1": {"word_types": 1.0, "mean_pieces_per_word": 3.0, "p95_pieces_per_word": 3.0}},
        common_long_words=[("commonlongword", 100, 1)],
    )

    text = report.read_text(encoding="utf-8")
    assert "Vocab size: `16000`" in text
    assert "Common Long Word Examples" in text
    assert "`commonlongword`" in text
