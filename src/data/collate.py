from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.translation_dataset import EncodedTranslationExample


@dataclass(frozen=True)
class TranslationBatch:
    src_ids: torch.Tensor
    tgt_in_ids: torch.Tensor
    tgt_out_ids: torch.Tensor
    src_key_padding_mask: torch.Tensor
    tgt_key_padding_mask: torch.Tensor
    tgt_causal_mask: torch.Tensor

    def to(self, device: torch.device) -> "TranslationBatch":
        return TranslationBatch(
            src_ids=self.src_ids.to(device, non_blocking=True),
            tgt_in_ids=self.tgt_in_ids.to(device, non_blocking=True),
            tgt_out_ids=self.tgt_out_ids.to(device, non_blocking=True),
            src_key_padding_mask=self.src_key_padding_mask.to(device, non_blocking=True),
            tgt_key_padding_mask=self.tgt_key_padding_mask.to(device, non_blocking=True),
            tgt_causal_mask=self.tgt_causal_mask.to(device, non_blocking=True),
        )


def make_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)


def _rounded_length(length: int, pad_to_multiple_of: int | None) -> int:
    if not pad_to_multiple_of or pad_to_multiple_of <= 1:
        return length
    return ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of


def _pad_sequences(sequences: list[list[int]], pad_id: int, pad_to_multiple_of: int | None) -> torch.Tensor:
    max_len = _rounded_length(max(len(seq) for seq in sequences), pad_to_multiple_of)
    output = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        output[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return output


class TranslationCollator:
    def __init__(self, pad_id: int = 0, pad_to_multiple_of: int | None = 8) -> None:
        self.pad_id = pad_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples: list[EncodedTranslationExample]) -> TranslationBatch:
        if not examples:
            raise ValueError("Cannot collate an empty batch.")
        src_ids = _pad_sequences([example.src_ids for example in examples], self.pad_id, self.pad_to_multiple_of)
        tgt_in_ids = _pad_sequences([example.tgt_in_ids for example in examples], self.pad_id, self.pad_to_multiple_of)
        tgt_out_ids = _pad_sequences([example.tgt_out_ids for example in examples], self.pad_id, self.pad_to_multiple_of)
        return TranslationBatch(
            src_ids=src_ids,
            tgt_in_ids=tgt_in_ids,
            tgt_out_ids=tgt_out_ids,
            src_key_padding_mask=src_ids.eq(self.pad_id),
            tgt_key_padding_mask=tgt_in_ids.eq(self.pad_id),
            tgt_causal_mask=make_causal_mask(tgt_in_ids.size(1)),
        )
