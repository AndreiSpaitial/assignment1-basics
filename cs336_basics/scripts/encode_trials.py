import argparse
import time

from cs336_basics.tokenizer import BPETokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that encodes a file with the given BPE checkpoint.")
    parser.add_argument("checkpoint_path", help="The first input string")
    parser.add_argument("input_file_tinystories", help="The second input string")
    parser.add_argument("input_file_owt", help="The second input string")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    input_file_tinystories = args.input_file_tinystories
    input_file_owt = args.input_file_owt

    bpe_tokenizer = BPETokenizer(
        special_tokens=["<|endoftext|>"],
        num_processes=64,
        checkpoint_dir=checkpoint_path,
        vocab_size=10_000,
    )
    bpe_tokenizer.load_checkpoint()

    def profile_tokens(input_file, name):
        with open(input_file) as f:
            text = f.read()

        encoding_start = time.time()
        tokens = bpe_tokenizer.encode(text)
        encoding_end = time.time()

        encoding_time = encoding_end - encoding_start

        raw_bytes = len(text.encode("utf-8"))
        compression_ratio = raw_bytes / len(tokens)

        compression_througput = raw_bytes/encoding_time

        print(f"{name} {compression_ratio=:}")
        print(f"{name} {compression_througput=:}")

    profile_tokens(input_file_tinystories, "TinyStories")
    profile_tokens(input_file_owt, "OWT")
