from collections.abc import Iterable

import torch


def gradient_clipping(params: Iterable[torch.nn.Parameter], M: float) -> None:
    eps = 1e-6
    total_norm = 0
    for p in params:
        if p.grad is None:
            continue

        total_norm += (p.grad.data**2).sum()

    total_norm = torch.sqrt(total_norm)

    if total_norm <= M:
        return

    for p in params:
        if p.grad is None:
            continue
        p.grad.data *= M / (total_norm + eps)
