import os
import typing

import torch
from torch import nn

from cs336_basics.dataloader import LLMDataLoader


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: LLMDataLoader,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    training_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "dataloader": dataloader.state_dict(),
        "iteration": iteration,
    }

    torch.save(training_dict, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: LLMDataLoader,
) -> int:
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    dataloader.load_state_dict(state_dict["dataloader"])

    return state_dict["iteration"]
