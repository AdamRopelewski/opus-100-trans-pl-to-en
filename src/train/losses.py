from __future__ import annotations

import torch
from torch import nn


class LabelSmoothedCrossEntropy(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, ignore_index: int = 0) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.size(-1)
        return self.loss(logits.reshape(-1, vocab_size), targets.reshape(-1))
