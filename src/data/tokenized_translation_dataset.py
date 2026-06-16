from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from src.data.translation_dataset import (
    EncodedTranslationExample,
    SpecialTokenIds,
    SplitLoadStats,
)


@dataclass(frozen=True)
class TokenizedSplitConfig:
    split: str
    paths: tuple[Path, ...]
    token_ids: SpecialTokenIds = SpecialTokenIds()
    shuffle_files: bool = False
    shuffle_buffer_size: int = 0
    seed: int = 42
    batch_size: int = 8192


class TokenizedTranslationDataset(IterableDataset[EncodedTranslationExample]):
    def __init__(self, config: TokenizedSplitConfig) -> None:
        self.config = config

    def __iter__(self) -> Iterator[EncodedTranslationExample]:
        worker = get_worker_info()
        paths = list(self.config.paths)
        rng = random.Random(self.config.seed + (worker.id if worker else 0))
        if self.config.shuffle_files:
            rng.shuffle(paths)
        if worker is not None:
            paths = paths[worker.id :: worker.num_workers]
        examples = _iter_tokenized_examples(
            paths, self.config.batch_size, self.config.token_ids
        )
        if self.config.shuffle_buffer_size > 1:
            yield from _shuffle_buffer(examples, self.config.shuffle_buffer_size, rng)
            return
        yield from examples


def _iter_tokenized_examples(
    paths: list[Path], batch_size: int, token_ids: SpecialTokenIds
) -> Iterator[EncodedTranslationExample]:
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            data = batch.to_pydict()
            for src_ids, tgt_ids in zip(
                data["src_ids"],
                data["tgt_ids"],
                strict=True,
            ):
                target_piece_ids = list(tgt_ids)
                yield EncodedTranslationExample(
                    src_ids=list(src_ids),
                    tgt_in_ids=[token_ids.bos_id] + target_piece_ids,
                    tgt_out_ids=target_piece_ids + [token_ids.eos_id],
                )


def _shuffle_buffer(
    examples: Iterator[EncodedTranslationExample], buffer_size: int, rng: random.Random
) -> Iterator[EncodedTranslationExample]:
    buffer: list[EncodedTranslationExample] = []
    for example in examples:
        if len(buffer) < buffer_size:
            buffer.append(example)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = example
    rng.shuffle(buffer)
    yield from buffer


def count_tokenized_rows(paths: tuple[Path, ...]) -> int:
    return sum(int(pq.read_metadata(path).num_rows) for path in paths)


def load_tokenized_translation_dataset(
    split: str,
    paths: tuple[Path, ...],
    *,
    token_ids: SpecialTokenIds = SpecialTokenIds(),
    shuffle_files: bool = False,
    shuffle_buffer_size: int = 0,
    seed: int = 42,
    batch_size: int = 8192,
) -> tuple[TokenizedTranslationDataset, SplitLoadStats]:
    if not paths:
        raise FileNotFoundError(f"No tokenized parquet files found for split '{split}'.")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Tokenized split file not found: {path}")
    rows = count_tokenized_rows(paths)
    return (
        TokenizedTranslationDataset(
            TokenizedSplitConfig(
                split=split,
                paths=paths,
                token_ids=token_ids,
                shuffle_files=shuffle_files,
                shuffle_buffer_size=shuffle_buffer_size,
                seed=seed,
                batch_size=batch_size,
            )
        ),
        SplitLoadStats(split=split, rows_in=rows, rows_out=rows, overlength_rows=0),
    )
