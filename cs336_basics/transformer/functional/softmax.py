import torch

from einops import reduce


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max, _ = x.max(dim=dim, keepdim=True)

    x_exp = torch.exp(x-x_max)
    x_norm = x_exp.sum(dim=dim, keepdim=True)

    return x_exp/x_norm
