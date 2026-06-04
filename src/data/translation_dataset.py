from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pyarrow.parquet as pq
from torch.utils.data import Dataset

from src.utils.tokenizer import iter_parallel_rows


@dataclass(frozen=True)
class SpecialTokenIds:
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3


@dataclass(frozen=True)
class EncodedTranslationExample:
    src_ids: list[int]
    tgt_in_ids: list[int]
    tgt_out_ids: list[int]


@dataclass(frozen=True)
class SplitLoadStats:
    split: str
    rows_in: int
    rows_out: int
    overlength_rows: int


class TranslationDataset(Dataset[EncodedTranslationExample]):
    def __init__(self, examples: list[EncodedTranslationExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedTranslationExample:
        return self.examples[index]


def _limit_rows(rows: Iterator[tuple[str, str]], limit_samples: int | None) -> Iterator[tuple[str, str]]:
    for index, row in enumerate(rows):
        if limit_samples is not None and index >= limit_samples:
            break
        yield row


def build_translation_examples(
    rows: Iterator[tuple[str, str]],
    encode: Callable[[str], list[int]],
    token_ids: SpecialTokenIds,
    max_seq_len: int,
    drop_overlength: bool = True,
) -> tuple[list[EncodedTranslationExample], int, int]:
    examples: list[EncodedTranslationExample] = []
    rows_in = 0
    overlength_rows = 0
    for pl_text, en_text in rows:
        rows_in += 1
        src_ids = encode(pl_text) + [token_ids.eos_id]
        target_piece_ids = encode(en_text)
        tgt_in_ids = [token_ids.bos_id] + target_piece_ids
        tgt_out_ids = target_piece_ids + [token_ids.eos_id]
        if max(len(src_ids), len(tgt_in_ids), len(tgt_out_ids)) > max_seq_len:
            overlength_rows += 1
            if drop_overlength:
                continue
        examples.append(EncodedTranslationExample(src_ids=src_ids, tgt_in_ids=tgt_in_ids, tgt_out_ids=tgt_out_ids))
    return examples, rows_in, overlength_rows


def load_translation_dataset(
    split: str,
    path: Path,
    encode: Callable[[str], list[int]],
    token_ids: SpecialTokenIds,
    max_seq_len: int,
    drop_overlength: bool = True,
    limit_samples: int | None = None,
) -> tuple[TranslationDataset, SplitLoadStats]:
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    rows = _limit_rows(iter_parallel_rows(path), limit_samples)
    examples, rows_in, overlength_rows = build_translation_examples(
        rows,
        encode,
        token_ids,
        max_seq_len=max_seq_len,
        drop_overlength=drop_overlength,
    )
    return (
        TranslationDataset(examples),
        SplitLoadStats(split=split, rows_in=rows_in, rows_out=len(examples), overlength_rows=overlength_rows),
    )


def count_parquet_rows(path: Path) -> int:
    return int(pq.read_metadata(path).num_rows)
