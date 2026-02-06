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
