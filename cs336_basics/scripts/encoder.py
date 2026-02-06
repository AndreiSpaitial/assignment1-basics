import argparse
import time

import numpy as np

from cs336_basics.tokenizer import BPETokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that encodes a file with the given BPE checkpoint.")
    parser.add_argument("checkpoint_path", help="The first input string")
    parser.add_argument("input_file", help="The second input string")
    parser.add_argument("output_file", help="The second input string")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    input_file = args.input_file
    output_file = args.output_file

    bpe_tokenizer = BPETokenizer(
        special_tokens=["<|endoftext|>"],
        num_processes=64,
        checkpoint_dir=checkpoint_path,
        vocab_size=10_000,
    )
    bpe_tokenizer.load_checkpoint()

    total_chunks = 0
    with open(input_file) as f:
        for line in f:
            total_chunks += 1

    with open(input_file) as f:
        tokens = bpe_tokenizer.encode_iterable(f, total=total_chunks)

        tokens_np = np.array(list(tokens), dtype=np.uint16)

    np.save(output_file, tokens_np)
