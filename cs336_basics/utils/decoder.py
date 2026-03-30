import torch
from torch import Tensor

from einops import reduce, pack
from jaxtyping import Float, Int64, Bool

from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.transformer import Transformer
from cs336_basics.transformer.functional import softmax


def sample_next_word(
    y: Float[Tensor, "b seq_len vocab_size"],
    temperature: float = 1.0,
    p: float = 0.4,
) -> Int64[Tensor, "b 1"]:
    last_token_logits: Float[Tensor, "b vocab_size"] = y[:, -1, :]
    last_token_logits /= temperature
    last_token_probs: Float[
        Tensor, "b vocab_size"
    ] = softmax(last_token_logits, dim=-1)

    values, indices = torch.sort(last_token_probs, descending=True)
    values_cumsum: Float[Tensor, "b vocab_size"] = torch.cumsum(values, -1)
    values_cumsum_select: Bool[Tensor, "b vocab_size"] = values_cumsum >= p
    values_cumsum_select = torch.roll(values_cumsum_select, shifts=1)
    values_cumsum_select[:, 0] = True
    values_cumsum_select = torch.gather(
        values_cumsum_select,
        dim=-1,
        index=indices.argsort()
    )
    last_token_probs[~values_cumsum_select] = 0
    last_token_probs /= reduce(
        last_token_probs,
        "b seq_len -> b 1",
        "sum"
    )

    return torch.multinomial(last_token_probs, num_samples=1)


def decode_batch(
    model: Transformer,
    x: Int64[Tensor, "b seq_len"],
    max_tokens: int = 10,
    temperature: float = 1.0,
    p: float = 0.4,
) -> Int64[Tensor, "b new_seq_len"]:
    for _ in range(max_tokens):
        y = model(x)
        next_token = sample_next_word(y, temperature, p)
        x, _ = pack([x, next_token], "b *")

    return x


def model_output_to_text(
    o: Int64[Tensor, "b seq_len"],
    tokenizer: BPETokenizer,
) -> list[str]:
    x = o.cpu().detach().numpy()
    ret = []
    for el in x:
        ret.append(tokenizer.decode(el))

    return ret


def run_user_prompt(
    model: Transformer,
    tokenizer: BPETokenizer,
    user_prompt: str,
    max_tokens: int = 10,
    temperature: float = 1.0,
    p: float = 0.4,
    termination_token: bytes = b"<|endoftext|>",
    device: str = "cpu",
) -> str:
    x: Int64[Tensor, "1 seq_len"] = torch.tensor(
        [tokenizer.encode(user_prompt)],
        device=device,
    )
    termination_token_id = tokenizer._dictionary_rev[termination_token]

    for _ in range(max_tokens):
        y: Float[Tensor, "1 seq_len vocab_size"] = model(x)
        next_token: Int64[
            Tensor, "1 seq_len_plus_1"
        ] = sample_next_word(y, temperature, p)
        if next_token[0, 0] == termination_token_id:
            break
        x, _ = pack([x, next_token], "b *")

    return model_output_to_text(x, tokenizer)[0]
