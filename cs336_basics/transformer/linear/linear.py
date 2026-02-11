import torch
import torch.nn as nn

from einops import einsum


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(LinearLayer, self).__init__()

        empty = torch.empty(
            out_features,
            in_features,
            dtype=dtype,
            device=device
        )

        self.W = nn.Parameter(empty)
        std = (2/(in_features+out_features))**0.5
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.W,
            "... in_features, out_features in_features -> ... out_features"
        )
