from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor

from einops import einsum, rearrange
from jaxtyping import Float

from cs336_basics.transformer.attention import masked_attention
from cs336_basics.transformer.linear import LinearLayer
from cs336_basics.transformer.positional import ROPE


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int = 1,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(MultiheadAttention, self).__init__()

        self.heads = heads
        d_k = d_v = d_model // heads
        self.d_k = d_k
        self.d_v = d_v

        self.rope: Callable = lambda x, *args, **kwargs: x
        if theta is not None:
            max_seq_len = max_seq_len or 128
            self.rope = ROPE(theta, d_k, max_seq_len=max_seq_len)

        self.W_QKV = LinearLayer(
            d_model,
            heads * (d_k + d_k + d_v),
            device,
            dtype
        )
        self.W_0 = LinearLayer(heads * d_v, d_model, device, dtype)

    def forward(
        self,
        x: Float[Tensor, "batch ... seq_len d_model"],
        token_positions: Float[Tensor, "batch ... seq_len"] | None = None,
    ) -> Float[Tensor, "batch ... seq_len d_model"]:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len)

        QKV: Float[Tensor, "batch ... seq_len d_hkkv"] = self.W_QKV(x)
        Q: Float[Tensor, "batch ... seq_len (h d_k)"] = QKV[
            ..., :(self.heads*self.d_k)
        ]
        K: Float[Tensor, "batch ... seq_len (h d_k)"] = QKV[
            ..., (self.heads*self.d_k):(2*self.heads*self.d_k)
        ]
        V: Float[Tensor, "batch ... seq_len (h d_v)"] = QKV[
            ..., (2*self.heads*self.d_k):
        ]

        Q = rearrange(
            Q,
            "batch ... seq_len (h d_k) -> batch ... h seq_len d_k",
            h=self.heads,
        )
        Q = self.rope(Q, token_positions)

        K = rearrange(
            K,
            "batch ... seq_len (h d_k) -> batch ... h seq_len d_k",
            h=self.heads,
        )
        K = self.rope(K, token_positions)

        V = rearrange(
            V,
            "batch ... seq_len (h d_v) -> batch ... h seq_len d_v",
            h=self.heads,
        )

        mask = (
            1. - torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        ).bool()

        ret: Float[Tensor, "batch ... h seq_len d_v"] = masked_attention(Q, K, V, mask)
        ret = rearrange(ret, "batch ... h seq_len d_v -> batch ... seq_len (h d_v)")

        ret = self.W_0(ret)
        return ret
