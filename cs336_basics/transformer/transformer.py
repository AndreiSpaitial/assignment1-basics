import torch
import torch.nn as nn
from torch import Tensor

from jaxtyping import Float, Int64

from cs336_basics.transformer import TransformerBlock
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.functional import softmax
from cs336_basics.transformer.linear import LinearLayer
from cs336_basics.transformer.positional import ROPE
from cs336_basics.transformer.rmsnorm import RMSNorm


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        heads: int = 1,
        d_ff: int | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
        rope: ROPE | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(Transformer, self).__init__()

        self.embedding = Embedding(
            vocab_size,
            d_model,
            device,
            dtype,
        )

        transformer_blocks = []
        for _ in range(num_layers):
            transformer_blocks.append(
                TransformerBlock(
                    d_model,
                    heads,
                    d_ff,
                    theta,
                    max_seq_len,
                    rope,
                    device,
                    dtype,
                )
            )
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.out_norm = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        self.out_projection = LinearLayer(
            d_model,
            vocab_size,
            device,
            dtype,
        )

    def forward(
        self, tokens: Int64[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        x: Float[Tensor, "batch seq_len d_model"] = self.embedding(tokens)
        x = self.transformer_blocks(x)
        x = self.out_norm(x)

        x_out: Float[Tensor, "batch seq_len vocab_size"] = self.out_projection(x)

        return x_out

    def flops(self, x: tuple[int, ...]) -> int:
        transformer_flops = sum(
            transformer_block.flops(x)
            for transformer_block in self.transformer_blocks
        )
        print(f"Total transformer blocks flops: {transformer_flops:_}")
        output_flops = self.out_projection.flops(x)
        print(f"Transformer output projection flops: {output_flops:_}")

        return transformer_flops + output_flops

    def num_params(self) -> int:
        return (
            sum(
                transformer_block.num_params()
                for transformer_block in self.transformer_blocks
            ) +
            self.out_projection.num_params()
        )
