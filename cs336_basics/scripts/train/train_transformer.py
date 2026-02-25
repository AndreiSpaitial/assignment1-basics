import os
from functools import partial

import torch
import numpy as np
import typer
import yaml

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cs336_basics.dataloader import LLMDataset, LLMDataLoader
from cs336_basics.optimizer import AdamW, cosine_annealing, gradient_clipping
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.transformer import Transformer
from cs336_basics.transformer.functional import ce_loss, perplexity
from cs336_basics.transformer.positional import ROPE
from cs336_basics.utils import save_checkpoint, load_checkpoint, decode_batch, model_output_to_text


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    with os.scandir(checkpoint_dir) as entries:
        # Extract names of files (ignoring directories)
        files = (entry.name for entry in entries if entry.is_file())
        max_cp: str | os.PathLike | None = max(files, default=None)

    if max_cp is None:
        return None

    checkpoint_path = os.path.join(checkpoint_dir, max_cp)

    return checkpoint_path


def make_transformer_lm(
    train_conf: dict, device_str: str
) -> tuple[Transformer, BPETokenizer]:
    device = torch.device(device_str)
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

    return Transformer(
        vocab_size,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope=rope,
        device=device,
    ), bpe_tokenizer


def make_optimizer(train_conf: dict) -> dict:
    optimizer_conf = train_conf["optimizer"]
    kw_args = {}

    if "lr" in optimizer_conf:
        kw_args["lr"] = optimizer_conf["lr"]
    if "weight_decay" in optimizer_conf:
        kw_args["weight_decay"] = optimizer_conf["weight_decay"]
    if "betas" in optimizer_conf:
        kw_args["betas"] = tuple(optimizer_conf["betas"])
    if "eps" in optimizer_conf:
        kw_args["eps"] = optimizer_conf["eps"]

    return kw_args


def compute_val_loss(
    model: Transformer, eval_dataloader: LLMDataLoader
) -> tuple[float, float]:
    total_loss = 0.
    total_perplexity = 0.
    eval_dataloader.set_for_epoch(0)
    with torch.no_grad():
        for x, y in tqdm(eval_dataloader, desc="Evaluating model"):
            y_pred = model(x)
            total_batch_loss = perplexity(y_pred, y, to_exp=False)
            total_loss += total_batch_loss[0]

            total_batch_perplexity = perplexity(y_pred, y)
            total_perplexity += total_batch_perplexity[0]

    return total_loss, total_perplexity


def main(
    config_path: str = "",
    device: str = "cpu",
):
    with open(config_path, "r") as f:
        train_conf = yaml.safe_load(f)

    train_dataset_npy = np.load(train_conf["train_path"], mmap_mode="r")
    eval_dataset_npy = np.load(train_conf["eval_path"], mmap_mode="r")[:2000]
    epochs = train_conf["epochs"]
    checkpoint_every = train_conf["checkpoint_every"]
    eval_every = train_conf["eval_every"]
    max_steps = train_conf.get("max_steps")
    writer = SummaryWriter('cs336_basics/scripts/train/runs/small_experiments/')

    train_dataset = LLMDataset(train_dataset_npy)
    eval_dataset = LLMDataset(eval_dataset_npy)

    train_data_loader = LLMDataLoader(
        train_dataset,
        train_conf["transformer"]["context_length"],
        train_conf["batch_size"],
        device,
        train_conf["shuffle"],
    )
    eval_data_loader = LLMDataLoader(
        eval_dataset,
        train_conf["transformer"]["context_length"],
        train_conf["batch_size"],
        device,
        train_conf["shuffle"],
    )

    transformer_lm, bpe_tokenizer = make_transformer_lm(
        train_conf,
        device,
    )

    eval_batch, _ = eval_data_loader._get_batch()
    eval_prompts = model_output_to_text(eval_batch, bpe_tokenizer)

    optimizer_args = make_optimizer(train_conf)

    optimizer = AdamW(
        transformer_lm.parameters(),
        **optimizer_args,
    )
    lr_scheduler = None
    lr_conf = train_conf.get("lr_conf")
    if lr_conf is not None:
        lr_scheduler = partial(
            cosine_annealing,
            lr_max=lr_conf["lr_max"],
            lr_min=lr_conf["lr_min"],
            t_W=lr_conf["t_W"],
            t_C=lr_conf["t_C"]
        )
    optimizer.lr_scheduler = lr_scheduler
    M = train_conf["gradient_clipping"]["M"]
    gradient_clipper = partial(gradient_clipping, M=M)
    optimizer.gradient_clipper = gradient_clipper

    global_iter = 0
    start_epoch = 0

    checkpoint_dir = train_conf["checkpoint_dir"]
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        global_iter, start_epoch = load_checkpoint(
            checkpoint_path,
            transformer_lm,
            optimizer,
            train_data_loader,
        )
    print(f"{len(train_data_loader):=}")
    epoch_iter = train_data_loader._i // train_data_loader.batch_size
    print(f"Reloaded {epoch_iter:=}, {global_iter:=}")
    running_loss = 0.

    for epoch in range(start_epoch, epochs):
        for x, y in tqdm(
                train_data_loader,
                initial=epoch_iter,
        ):
            optimizer.zero_grad()

            y_pred = transformer_lm(x)
            loss = ce_loss(y_pred, y)

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(
                "Hyperparameters/learning_rate",
                current_lr,
                global_iter,
            )
            global_iter += 1

            if global_iter and global_iter % eval_every == 0:
                print("Evaluating model")
                print(f"Training loss: {running_loss / eval_every}")
                writer.add_scalar(
                    "Loss/train",
                    running_loss / eval_every,
                    global_iter,
                )
                eval_loss, eval_perplexity = compute_val_loss(
                    transformer_lm,
                    eval_data_loader,
                )
                writer.add_scalar(
                    "Loss/eval",
                    eval_loss,
                    global_iter,
                )
                print(f"Validation loss: {eval_loss}")
                print(f"Validation perplexity: {eval_perplexity}")
                writer.add_scalar(
                    "Perplexity/eval",
                    eval_perplexity,
                    global_iter,
                )

                print("Running eval set prompts")
                eval_set_output = decode_batch(
                    transformer_lm,
                    eval_batch
                )
                eval_set_text = model_output_to_text(
                    eval_set_output,
                    bpe_tokenizer
                )
                print("Done running prompts")
                for prompt_i, output in enumerate(eval_set_text):
                    writer.add_text(f"Samples/Prompt_{prompt_i}",
                                    f"""
                                    Input: {eval_prompts[prompt_i]} \n
                                    Output: {output} \n
                                    """,
                                    global_iter,
                                    )

                running_loss = 0.

            if global_iter and global_iter % checkpoint_every == 0:
                checkpoint_name = f"checkpoint_{global_iter:06}.pth"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                save_checkpoint(
                    transformer_lm,
                    optimizer,
                    train_data_loader,
                    global_iter,
                    checkpoint_path,
                    epoch
                )
            if max_steps and global_iter > max_steps:
                raise ValueError("stawwwwwp")


if __name__ == "__main__":
    typer.run(main)
