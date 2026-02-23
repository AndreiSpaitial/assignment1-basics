from collections.abc import Iterator

import numpy as np
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

        self._i = self._i + 1

        x = np.array(x, dtype=np.int16)
        y = np.array(y, dtype=np.int16)

        return torch.tensor(x).to(self.device), torch.tensor(y).to(self.device)

    def set_for_epoch(self, epoch: int):
        self._i = 0
        last_ind = len(self.llm_dataset) - self.context_length
        self._indices = torch.randperm(last_ind)

    def __iter__(self) -> Iterator[
        tuple[
            Int16[Tensor, "b context_length"],
            Int16[Tensor, "b context_length"],
        ]
    ]:
        for _ in range(self._i, len(self._indices)):
            yield self._get_batch()

    def __next__(self) -> tuple[
        Int16[Tensor, "b context_length"],
        Int16[Tensor, "b context_length"],
    ]:
        return self._get_batch()

    def __len__(self) -> int:
        return len(self._indices)

    def state_dict(self) -> dict:
        return {
            "indices": self._indices,
            "i": self._i,
        }

    def load_state_dict(self, state_dict: dict):
        self._indices = state_dict["indices"]
        self._i = state_dict["i"]
