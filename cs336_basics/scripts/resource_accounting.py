def account_resources(b, context_length, d_model, d_ff, num_heads, vocab_size, num_layers):
    def to_gb(x):
        return (x * 4) / (1024**3)

    transformer_block_p = (d_model) + (3 * d_model**2 + d_model**2) + (3 * d_model * d_ff)
    transformer_block_o = 3 * ((d_model) + (3 * d_model**2 + d_model**2) + (3 * d_model * d_ff))
    transformer_block_a = (b * context_length * d_model) + (3 * b * context_length * d_model + 2 * b * num_heads * context_length**2 + b * context_length * d_model) + (2 * b * context_length * d_ff + b * context_length * d_model)

    total = (
        num_layers * (transformer_block_p + transformer_block_o + transformer_block_a) +
        (d_model + 3 * d_model + b * context_length * vocab_size) +
        (d_model * vocab_size + 3 * d_model * vocab_size + b * context_length * vocab_size) +
        (b * context_length)
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
          a: {b * context_length * d_model=:_} ({to_gb(b * context_length * d_model):.4f} GiB)

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
      a: {b * context_length * vocab_size=:_} ({to_gb(b * context_length * vocab_size):.4f} GiB)
    • cross-entropy on logits:
      a: {b * context_length=:_} ({to_gb(b * context_length):.4f} GiB)
    TOTAL: {total:_} ({to_gb(total):.4f} GiB)
    """


if __name__ == "__main__":
    print(
        account_resources(
            b=4,
            context_length=1024,
            d_model=1600,
            d_ff=6400,
            num_heads=25,
            num_layers=48,
            vocab_size=50257,
        )
    )
