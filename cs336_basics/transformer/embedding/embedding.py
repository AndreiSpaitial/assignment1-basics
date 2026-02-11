import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        device=None,
        dtype=None
    ):
        super(Embedding, self).__init__()

        empty = torch.empty(
            num_embeddings,
            embedding_dim,
            dtype=dtype,
            device=device,
        )

        self.embeddings = nn.Parameter(empty)
        nn.init.trunc_normal_(self.embeddings, mean=0, std=1, a=-3, b=3)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        return self.embeddings[x]
