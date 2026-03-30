import os

import torch
import typer
import yaml

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.transformer import Transformer
from cs336_basics.transformer.positional import ROPE
from cs336_basics.utils import run_user_prompt


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    with os.scandir(checkpoint_dir) as entries:
        # Extract names of files (ignoring directories)
        files = (entry.name for entry in entries if entry.is_file())
        max_cp: str | os.PathLike | None = max(files, default=None)

    if max_cp is None:
        return None

    checkpoint_path = os.path.join(checkpoint_dir, max_cp)

    return checkpoint_path


def main(
    train_conf_path: str = "",
    model_checkpoint_path: str | None = None,
    max_tokens_per_prompt: int = 10,
    temperature: float = 1.0,
    p: float = 0.4,
    max_tokens: int = 10,
    device: str = "cpu",
):
    with open(train_conf_path, "r") as f:
        train_conf = yaml.safe_load(f)

    device = torch.device(device)
    tokenizer_conf = train_conf["tokenizer"]
    bpe_tokenizer = BPETokenizer(
        special_tokens=tokenizer_conf["special_tokens"],
        num_processes=256,
        checkpoint_dir=tokenizer_conf["checkpoint_dir"],
        vocab_size=10_000,
    )
    bpe_tokenizer.load_checkpoint()

    transformer_conf = train_conf["transformer"]
    d_model = transformer_conf["d_model"]
    num_heads = transformer_conf["num_heads"]
    context_length = transformer_conf["context_length"]
    d_model = transformer_conf["d_model"]
    num_layers = transformer_conf["num_layers"]
    d_ff = transformer_conf["d_ff"]

    vocab_size = bpe_tokenizer.vocab_size
    rope = ROPE(
        train_conf["rope"]["theta"],
        d_model // num_heads,
        max_seq_len=context_length,
        device=device,
    )

    model = Transformer(
        vocab_size,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope=rope,
        device=device,
    )

    if model_checkpoint_path is None:
        checkpoint_dir = train_conf["checkpoint_dir"]
        model_checkpoint_path = find_latest_checkpoint(checkpoint_dir)

    if model_checkpoint_path is not None:
        print(f"Loading model checkpoint from {model_checkpoint_path}")
        state_dict = torch.load(model_checkpoint_path)
        model.load_state_dict(state_dict["model"])

    prompt = ""
    while prompt != "quit":
        prompt = input("Enter your prompt: ")
        if prompt == "quit":
            break

        ret = run_user_prompt(
            model,
            bpe_tokenizer,
            prompt,
            temperature=temperature,
            p=p,
            max_tokens=max_tokens,
            device=device
        )
        print(ret)


if __name__ == "__main__":
    typer.run(main)
