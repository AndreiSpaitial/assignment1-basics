import torch
import torch.nn as nn

from einops import reduce


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super(RMSNorm, self).__init__()

        self.d_model = d_model
        self.eps = eps

        self.g = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = reduce(x**2, "... d_model -> ... 1", "mean") + self.eps
        rms = torch.sqrt(rms)

        res = x/rms * self.g

        return res.to(in_dtype)

    def flops(self, x: tuple[int, ...]) -> int:
        n_tokens = 1

        for d in x[:-1]:
            n_tokens *= d

        return n_tokens * x[-1]

    def num_params(self):
        return self.d_model
