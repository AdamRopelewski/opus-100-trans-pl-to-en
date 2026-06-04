from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_inverse_sqrt_scheduler(optimizer: Optimizer, warmup_steps: int) -> LambdaLR:
    warmup = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        current_step = max(1, step)
        return min(current_step ** -0.5, current_step * (warmup ** -1.5)) * math.sqrt(warmup)

    return LambdaLR(optimizer, lr_lambda)
