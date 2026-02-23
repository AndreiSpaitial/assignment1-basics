from collections.abc import Iterable

import torch
from torch import Tensor
from jaxtyping import Int16

from cs336_basics.dataloader import LLMDataset


class LLMDataLoader:
    def __init__(
        self,
        dataset: LLMDataset,
        context_length: int,
        batch_size: int,
        device: str,
        shuffle: bool = False,
    ):
        self.llm_dataset = dataset
        self.context_length = context_length
        self.batch_size = batch_size
        self._i = 0
        self.device = device

        last_ind = len(self.llm_dataset) - self.context_length
        if shuffle:
            self._indices = torch.randperm(last_ind)
        else:
            self._indices = torch.arange(last_ind)

    def _get_batch(self) -> tuple[
        Int16[Tensor, "b context_length"],
        Int16[Tensor, "b context_length"]
    ]:
        x = []
        y = []

        for i in range(self._i, self.batch_size):
            x_start = self._indices[i]
            x_end = self._indices[i] + self.context_length
            x.append(self.llm_dataset[x_start:x_end])
            y.append(self.llm_dataset[(x_start+1):(x_end+1)])

        self._i = (self._i + 1) % len(self._indices)

        return torch.tensor(x).to(self.device), torch.tensor(y).to(self.device)

    def __iter__(self) -> Iterable[
        tuple[
            Int16[Tensor, "b context_length"],
            Int16[Tensor, "b context_length"],
        ]
    ]:
        for _ in self._indices:
            yield self._get_batch()
