## Problem 1
(a) What Unicode character does chr(0) return?
'\x00'
(b) How does this character’s string representation (__repr__()) differ from its printed representa-
tion?
print is empty, repr shows a string representation of its byte contents
(c) What happens when this character occurs in text? It may be helpful to play around with the
following in your Python interpreter and see if it matches your expectations:
```
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
The string object has an all-zero byte wherever `chr(0)` appears, but this byte is not displayed in a human-readable form, such as via `print`.

## Problem 2

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.

The range of possible values for utf-16 (0-2^16) and utf-32 (0-2^32) are much larger than utf-8 (0-2^8), making algorithms such as BPE less efficient. Also, it seems encoding to utf-16 and utf-32 also uses up more bytes for the same underlying string, further reducing efficiency of processing these strings.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into
a Unicode string. Why is this function incorrect? Provide an example of an input byte string
that yields incorrect results.
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'

Example: "'hello! こんにちは!'", and it fails because it doesn't properly handle characters represented by more than 1 byte.

(c) Give a two byte sequence that does not decode to any Unicode character(s).
"b'\xe3\x81'"

These are just the first two bytes of "は", and they don't make sense without the third byte


## Problem (train_bpe_tinystories): BPE Training on TinyStories

(a) ' responsibility', and it makes sense
(b) Pre-tokenising the raw corpus takes the longest

## Problem (train_bpe_owt): BPE Training on OWT

(a) b'----------------------------------------------------------------', not useful could probable do with some cleaning

(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
Deliverable: A one-to-two sentence response.


## Tokenizer experiments

(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyS-
tories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these
sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?

TinyStoriesEncoder:
  * TinyStories: 2.08 bytes/token
  * OWT: 1.7 bytes/token

OWTEncoder:
  * TinyStories: 1.95 bytes/token
  * OWT: 2.15 bytes/token


(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Com-
pare the compression ratio and/or qualitatively describe what happens.

TinyStoriesEncoder is less efficient at compressing OWT. (3.46 bytes/token vs 4.086 bytes/token)

(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to
tokenize the Pile dataset (825GB of text)?

TinyStories: 5M bytes/second -> ~45 hours for Pile
OWT: 5M bytes/second -> ~45 hours for Pile

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and devel-
opment datasets into a sequence of integer token IDs. We’ll use this later to train our language
model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is
uint16 an appropriate choice?

uint16 is fine because it is big enough to represent all our token ids ints (up to 32_000)

## Problem (transformer_accounting): Transformer LM resource accounting (5 points)
(a) Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
27
num_heads : 25
d_ff : 6,400
Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?

~2B parameters -> 4 * 2B bytes -> ~8GB of memory

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped
model. How many FLOPs do these matrix multiplies require in total? Assume that our input
sequence has context_length tokens.

Total transformer blocks flops: 3_845_337_907_200
Transformer output projection flops: 164_682_137_600
Total LM 4_010_020_044_800
Single transformer block
In projection flops: 5_242_880_000
Attention flops: 6_710_886_400
 Q.T @ K flops:             3_355_443_200             
 ROPE flops             0             
 softmax @ V flops             3_355_443_200             
Out projection flops: 5_242_880_000
MHA flops: 17_196_646_400
FFN flops: 62_914_560_000
Total flops for transformer block: 80_111_206_400

(c) Based on your analysis above, which parts of the model require the most FLOPs?
Seems like the FFN from the transformer block.

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24
layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the
model size increases, which parts of the Transformer LM take up proportionally more or less of
the total FLOPs?

GPT small:
Total transformer blocks flops: 430_033_600_512
Transformer output projection flops: 79_047_426_048
Total LM 509_081_026_560
Single transformer block
In projection flops: 1_207_959_552
Attention flops: 3_221_225_472
 Q.T @ K flops:             1_610_612_736             
 ROPE flops             0             
 softmax @ V flops             1_610_612_736             
Out projection flops: 1_207_959_552
MHA flops: 5_637_144_576
FFN flops: 30_198_988_800
Total flops for transformer block: 35_836_133_376

GPT medium:
Total transformer blocks flops: 1_172_526_071_808
Transformer output projection flops: 105_396_568_064
Total LM 1_277_922_639_872
Single transformer block
In projection flops: 2_147_483_648
Attention flops: 4_294_967_296
 Q.T @ K flops:             2_147_483_648             
 ROPE flops             0             
 softmax @ V flops             2_147_483_648             
Out projection flops: 2_147_483_648
MHA flops: 8_589_934_592
FFN flops: 40_265_318_400
Total flops for transformer block: 48_855_252_992

GPT large:
Total transformer blocks flops: 2_246_804_766_720
Transformer output projection flops: 131_745_710_080
Total LM 2_378_550_476_800
Single transformer block
In projection flops: 3_355_443_200
Attention flops: 5_368_709_120
 Q.T @ K flops:             2_684_354_560             
 ROPE flops             0             
 softmax @ V flops             2_684_354_560             
Out projection flops: 3_355_443_200
MHA flops: 12_079_595_520
FFN flops: 50_331_648_000
Total flops for transformer block: 62_411_243_520

Overall it seems the dominating operation is still the transformer block FFN, and it does not substantially decrease with lower model size. The most saved-on operation with decreasing model size is the attention operation, but it is still small compared to the FFN.


(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one
forward pass change? How do the relative contribution of FLOPs of the model components
change?

Total transformer blocks flops: 138_834_817_843_200
Transformer output projection flops: 2_634_914_201_600
Total LM 141_469_732_044_800
Single transformer block
In projection flops: 83_886_080_000
Attention flops: 1_717_986_918_400
 Q.T @ K flops:             858_993_459_200             
 ROPE flops             0             
 softmax @ V flops             858_993_459_200             
Out projection flops: 83_886_080_000
MHA flops: 1_885_759_078_400
FFN flops: 1_006_632_960_000
Total flops for transformer block: 2_892_392_038_400

The total number of flops went up substantially, and it seems the attention mechanism now dominates flops inside a transformer block.


## Problem (learning_rate_tuning): Tuning the learning rate (1 point)
As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s
see that in practice in our toy example. Run the SGD example above with three other values for the
learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each
of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of
training)?

With lr=1e1 it seems the loss converges quickly to zero, whereas with 1e2 it doesn't change and with 1e3 it quickly goes to infinity.


## Problem (adamwAccounting): Resource accounting for training with AdamW

(a) How much peak memory does running AdamW require? Decompose your answer based on the
    memory usage of the parameters, activations, gradients, and optimizer state. Express your answer
    in terms of the batch_size and the model hyperparameters (vocab_size, context_length,
    num_layers, d_model, num_heads). Assume d_ff = 4 ×d_model.
    see `cs336_basics/scripts/resource_accounting.py`

(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on
the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?
Deliverable: An expression that looks like a ·batch_size + b for numerical values a, b, and a
number representing the maximum batch size.
  ~ 14*batch_size + 30
  max batch_size ~= 3

(c) How many FLOPs does running one step of AdamW take?
~ 6 x n_tokens x num_params
  Forward pass: 2 x n_tokens x num_params
  Backward pass: 2 x 2 x (n_tokens x num_params)

(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second)
relative to the hardware’s theoretical peak FLOP throughput [Chowdhery et al., 2022]. An
NVIDIA A100 GPU has a theoretical peak of 19.5 teraFLOP/s for float32 operations. Assuming
you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a
batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022],
assume that the backward pass has twice the FLOPs of the forward pass.

Approx 23B params for GPT-XL

batch_size = 1024
Training FLOPs per batch: 6 * 1024 * 23B = 141T
MFU 50% -> ~10TFLOP/s
Training step time: 14s

400K steps -> 400K * 14s -> approx 64 days
