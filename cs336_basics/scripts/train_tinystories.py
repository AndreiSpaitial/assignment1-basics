from cs336_basics.tokenizer import BPETokenizer


if __name__ == "__main__":
    bpe_tokenizer = BPETokenizer(
        special_tokens=["<|endoftext|>"],
        num_processes=64,
        checkpoint_dir="cs336_basics/tokenizer/checkpoints/tinystories/",
        vocab_size=10_000,
    )

    bpe_tokenizer.train(
        "data/TinyStoriesV2-GPT4-train.txt",
        save_every=5000,
    )

    bpe_tokenizer.save_checkpoint()

    max_lengths = {
        len(word): word for _, word in bpe_tokenizer.dictionary.items()
    }

    print(f"Longest token in tinystories is {max_lengths[max(max_lengths.keys())]!r}")
