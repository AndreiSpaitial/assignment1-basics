import argparse

import yaml


def account_resources(b, context_length, d_model, d_ff, num_heads, vocab_size, num_layers):
    def to_gb(x):
        return (x * 4) / (1024**3)

    transformer_block_p = (
            2 * d_model  # RMSNorm params
        ) + (
            3 * d_model**2 +  # W_KQV projection
            d_model**2  # W_O projection
        ) + (
            d_model * d_ff +  # W1
            d_model * d_ff +  # W3
            d_ff * d_model   # W2
        )
    transformer_block_o = 3 * transformer_block_p
    transformer_block_a = (
        b * context_length * d_model +  # input (needed for residual)
        b * context_length * d_model  # RMSNorm1 activations
    ) + (
        b * context_length * d_model  # projection W_QKV activations
        ) + (
            3 * b * context_length * d_model +  # Q,K,V activations
            b * num_heads * context_length**2 +  # Q.T @ K activations
            b * num_heads * context_length**2 +  # softmax activations
            b * context_length * d_model  # softmax(Q.T @ V) activations
        ) + (
            b * context_length * d_model  # W_O projections
        ) + (
            b * context_length * d_model  # RMSNorm2 activations
        ) + (
            b * context_length * d_ff +  # W1 activations
            b * context_length * d_ff +  # W3 activations
            b * context_length * d_ff +  # Swish activations
            b * context_length * d_model  # W2 activations
        )

    total_p = (
        num_layers * transformer_block_p +
        d_model * vocab_size
    )
    total = (
        (
            4 * vocab_size * d_model +  # embeddings params + optimizer states
            b * context_length * d_model) +  # input sequence activations
            num_layers * (
                transformer_block_p + transformer_block_o + transformer_block_a
            ) +
        (
            4 * d_model * vocab_size +  # output projections params and optimizer states
            2 * b * context_length * vocab_size) +  # Output projection activations and their exp
            (2 * b * context_length)  # CE loss activations
    )

    return f"""
    • Transformer block:
        – Total for all transfomer blocks: {num_layers * (transformer_block_p + transformer_block_o + transformer_block_a):_} ({to_gb(num_layers * (transformer_block_p + transformer_block_o + transformer_block_a)):.4f} GiB)
        – TOTAL PER TRANSFORMER BLOCK:
          p_total: {transformer_block_p:_} ({to_gb(transformer_block_p):.4f} GiB)
          o_total: {transformer_block_o:_} ({to_gb(transformer_block_o):.4f} GiB)
          a_total: {transformer_block_a:_} ({to_gb(transformer_block_a):.4f} GiB)
        – RMSNorm(s)
          p: {d_model=:_} ({to_gb(d_model):.4f} GiB)
          o: {3 * d_model=:_} ({to_gb(3 * d_model):.4f} GiB)
          a: {2 * b * context_length * d_model=:_} ({2 * to_gb(b * context_length * d_model):.4f} GiB)

        – Multi-head self-attention sublayer: QKV projections, Q⊤K matrix multiply, softmax, 
          weighted sum of values, output projection.
          p:
            W_QKV: {3 * d_model * d_model=:_} ({to_gb(3 * d_model * d_model):.4f} GiB)
            W_O: {d_model * d_model=:_} ({to_gb(d_model * d_model):.4f} GiB)
          o:
            W_QKV: {3 * (3 * d_model * d_model)=:_} ({to_gb(9 * d_model * d_model):.4f} GiB)
            W_O: {3 * d_model * d_model=:_} ({to_gb(3 * d_model * d_model):.4f} GiB)
          a:
            Q, K, V: {3 * b * context_length * d_model=:_} ({to_gb(3 * b * context_length * d_model):.4f} GiB)
            (Q.T @ K), softmax: {2 * b * num_heads * context_length * context_length=:_} ({to_gb(2 * b * num_heads * context_length * context_length):.4f} GiB)
            softmax(Q.T @ K) @ V: {b * context_length * d_model=:_} ({to_gb(b * context_length * d_model)} GiB)
            V @ W_O: {b * context_length * d_model=:_} ({to_gb(b * context_length * d_model):.4f} GiB)

        – Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
          p:
            W_1: {d_model * d_ff=:_} ({to_gb(d_model * d_ff):.4f} GiB)
            W_3: {d_model * d_ff=:_} ({to_gb(d_model * d_ff):.4f} GiB)
            W_2: {d_ff * d_model=:_} ({to_gb(d_ff * d_model):.4f} GiB)
          o:
            W_1: {3 * d_model * d_ff=:_} ({to_gb(3 * d_model * d_ff):.4f} GiB)
            W_3: {3 * d_model * d_ff=:_} ({to_gb(3 * d_model * d_ff):.4f} GiB)
            W_2: {3 * d_ff * d_model=:_} ({to_gb(3 * d_ff * d_model):.4f} GiB)
          a:
            W_1, W_3: {2 * b * context_length * d_ff=:_} ({to_gb(2 * b * context_length * d_ff):.4f} GiB)
            W_2: {b * context_length * d_model=:_} ({to_gb(b * context_length * d_model):.4f} GiB)
    • final RMSNorm
      g: {d_model=:_} ({to_gb(d_model):.4f} GiB)
      o: {3 * d_model=:_} ({to_gb(3 * d_model):.4f} GiB)
      a: {b * context_length * d_model=:_} ({to_gb(b * context_length * d_model):.4f} GiB)
    • output embedding:
      g: {d_model * vocab_size=:_} ({to_gb(d_model * vocab_size):.4f} GiB)
      o: {3 * d_model * vocab_size=:_} ({to_gb(3 * d_model * vocab_size):.4f} GiB)
      a: {2 * b * context_length * vocab_size=:_} ({to_gb(b * context_length * vocab_size):.4f} GiB)
    • cross-entropy on logits:
      a: {b * context_length=:_} ({2* to_gb(b * context_length):.4f} GiB)
    TOTAL params: {total_p:_} ({to_gb(total_p):.4f} GiB)
    TOTAL: {total:_} ({to_gb(total):.4f} GiB)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-conf", type=str, help="Training config")
    parser.add_argument("--vocab-size", type=int, help="Vocab size to compute")

    args = parser.parse_args()
    train_conf_path = args.train_conf
    vocab_size = args.vocab_size
    with open(train_conf_path) as f:
        train_conf = yaml.safe_load(f)

    b = train_conf["batch_size"]
    transformer_conf = train_conf["transformer"]
    d_model = transformer_conf["d_model"]
    num_heads = transformer_conf["num_heads"]
    context_length = transformer_conf["context_length"]
    d_model = transformer_conf["d_model"]
    num_layers = transformer_conf["num_layers"]
    d_ff = transformer_conf["d_ff"]

    print(
        account_resources(
            b=b,
            context_length=context_length,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_layers=num_layers,
            vocab_size=vocab_size,
        )
    )
