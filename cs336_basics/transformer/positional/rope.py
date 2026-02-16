import torch
import torch.nn as nn
from torch import Tensor

from einops import einsum, rearrange
from jaxtyping import Float


class ROPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super(ROPE, self).__init__()

        # conceptually holds block diagonals for i=1...max_seq_len
        R: Float[Tensor, "seq d_k_2 2 2"] = (
            torch.zeros(
                max_seq_len, d_k//2, 2, 2,
                device=device,
            )
        )

        for i in range(max_seq_len):
            for k in range(d_k//2):
                # Formula has k as 1-based
                theta_ik = Tensor([i/theta**((2*(k+1)-2)/d_k)])
                block = torch.tensor([
                    [torch.cos(theta_ik), -torch.sin(theta_ik)],
                    [torch.sin(theta_ik), torch.cos(theta_ik)]
                ])

                R[i, k] = block

        self.register_buffer("R", R, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        R_seq: Float[Tensor, "... seq d_k_2 2 2"] = self.R[token_positions]
        x_splat = rearrange(
            x, "... seq (d_k_2 rot) -> ... seq d_k_2 rot", rot=2
        )

        ret = einsum(
            x_splat,
            R_seq,
            "... seq d_k_2 rot, ... seq d_k_2 rot1 rot -> ... seq d_k_2 rot1"
        )
        ret = rearrange(
            ret,
            "... seq d_k_2 rot -> ... seq (d_k_2 rot)",
            rot=2
        )

        return ret
