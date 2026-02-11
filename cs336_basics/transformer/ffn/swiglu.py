import math

import torch
import torch.nn as nn

from cs336_basics.transformer.linear import LinearLayer
from cs336_basics.transformer.functional import silu


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(SwiGLU, self).__init__()

        if d_ff is None:
            d_ff = int(math.floor(8/3 * d_model))
            d_ff = d_ff - (d_ff % 64)

        self.W1 = LinearLayer(d_model, d_ff, device, dtype)
        self.W3 = LinearLayer(d_model, d_ff, device, dtype)
        self.W2 = LinearLayer(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.W1(x)
        w3_x = self.W3(x)

        glu = silu(w1_x)*w3_x

        return self.W2(glu)
