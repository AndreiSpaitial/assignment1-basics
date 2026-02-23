import torch
from torch import Tensor

from einops import reduce, rearrange
from jaxtyping import Float, Int64

from cs336_basics.transformer.functional import ce_loss


def perplexity(
    o: Float[Tensor, "... seq_len vocab_size"],
    y: Float[Int64,  "... seq_len"],
) -> Float[Tensor, "1"]:
    l: Float[Tensor, "b ... seq_len"] = ce_loss(o, y, None)

    perplexity_seq = reduce(
        l,
        "b ... seq_len -> b ...",
        "mean"
    )

    perplexity_total = reduce(
        perplexity_seq,
        "b ... -> 1",
        "sum"
    )

    return perplexity_total
