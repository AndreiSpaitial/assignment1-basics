import torch
import torch.nn as nn
from torch import Tensor

from jaxtyping import Float

from cs336_basics.linear import LinearLayer


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        heads: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(MultiheadAttention, self).__init__()

        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = LinearLayer(d_model, heads * d_k, device, dtype)
        self.W_K = LinearLayer(d_model, heads * d_k, device, dtype)
        self.W_V = LinearLayer(d_model, heads * d_v, device, dtype)
        self.W_0 = LinearLayer(heads * d_v, d_model)

    def forward(
        self, x: Float[Tensor, "batch ... seq_len d_model"]
    ) -> Float[Tensor, "batch ... seq_len d_model"]:

        return None
