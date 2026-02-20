import torch
from torch import Tensor

from einops import reduce, rearrange
from jaxtyping import Float, Int64


def ce_loss(
    o: Float[Tensor, "... vocab_size"],
    y: Float[Int64,  "..."],
) -> Float[Tensor, "1"]:
    y = rearrange(y, "... -> ... 1")
    o -= reduce(
        o, "... vocab_size -> ... 1", "max"
    )
    o_y: Float[Tensor, "... 1"] = torch.take_along_dim(o, y, -1)

    exp_sum = torch.log(reduce(
        torch.exp(o),
        "... vocab_size -> ... 1",
        "sum"
    ))

    ret = -o_y + exp_sum
    ret = reduce(
        ret,
        "... -> 1",
        "mean",
    )

    return ret
