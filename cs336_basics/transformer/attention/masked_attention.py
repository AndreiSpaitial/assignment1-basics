import math

import torch
from torch import Tensor

from einops import einsum, rearrange
from jaxtyping import Float, Bool

from cs336_basics.transformer.functional import softmax


def masked_attention(
    Q: Float[Tensor, "batch ... seq_len_q d_k"],
    K: Float[Tensor, "batch ... seq_len_ref d_k"],
    V: Float[Tensor, "batch ... seq_len_ref d_v"],
    mask: Bool[Tensor, "seq_len_q seq_len_ref"] | None = None,
) -> Float[Tensor, "batch ... seq_len_q d_v"]:
    d_k = Q.shape[-1]
    Q_T = einsum(
        Q,
        K,
        "batch ... seq_len_q d_k, batch ... seq_len_ref d_k -> batch ... seq_len_q seq_len_ref"
    )
    Q_T /= math.sqrt(d_k)
    if mask is not None:
        Q_T[~mask] = -torch.inf

    Q_T = softmax(Q_T, dim=-1)
    ret = einsum(
        Q_T,
        V,
        "batch ... seq_len_q seq_len_ref, batch ... seq_len_ref d_v -> batch ... seq_len_q d_v"
    )

    return ret
