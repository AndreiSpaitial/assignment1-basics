import torch
import torch.nn as nn
from torch import Tensor

from jaxtyping import Float

from cs336_basics.transformer.attention import MultiheadAttention
from cs336_basics.transformer.ffn import SwiGLU
from cs336_basics.transformer.positional import ROPE
from cs336_basics.transformer.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int = 1,
        d_ff: int | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
        rope: ROPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(TransformerBlock, self).__init__()

        self.mha_norm = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
        self.mha = MultiheadAttention(
            d_model,
            heads,
            theta,
            max_seq_len,
            rope,
            device,
            dtype,
        )
        self.ff_norm = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
        self.ff = SwiGLU(
            d_model,
            d_ff,
            device,
            dtype,
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        x_1: Float[Tensor, "batch seq_len d_model"] = self.mha_norm(x)
        x_1 = self.mha(x_1)

        x_1 += x
        x_out = self.ff_norm(x_1)
        x_out = self.ff(x_out)

        return x_1 + x_out

    def flops(self, x: tuple[int, ...], detailed: bool = False) -> int:
        mha_flops = self.mha.flops(x, detailed)
        ffn_flops = self.ff.flops(x)

        if detailed:
            print(f"MHA flops: {mha_flops:_}")
            print(f"FFN flops: {ffn_flops:_}")

        return mha_flops + ffn_flops

    def num_params(self) -> int:
        return (
            self.mha_norm.num_params() +
            self.mha.num_params() +
            self.ff_norm.num_params() +
            self.ff.num_params()
        )
