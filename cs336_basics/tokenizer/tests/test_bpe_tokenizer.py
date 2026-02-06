import tempfile
from functools import partial

from collections import Counter, defaultdict

from cs336_basics.tokenizer import BPETokenizer


def _reverse_pair_index(token_to_pairs):
    ret = defaultdict(Counter)
    for pretoken, pairs in token_to_pairs.items():
        for pair, freq in pairs.items():
            ret[pair][pretoken] = freq

    return ret


EXPECTED_MERGES = [
    (b'a', b'b'),
    (b'k', b'ab'),
    (b'kab', b'h'),
    (b' ', b'ab'),
    (b'kabh', b'kabh'),
    (b'x', b'z'),
    (b'x', b'y'),
    (b'b', b'xz'),
    (b' ab', b'xy'),
    (b' ab', b'c'),
    (b' ', b'kabhkabh'),
    (b' ', b'bxz'),
    (b'x', b'b'),
    (b'xb', b'c'),
    (b'x', b'xbc'),
    (b'kabh', b'b'),
    (b'kabhb', b'h'),
    (b'kabhbh', b'kabh'),
    (b' ', b'xxbc'),
    (b' ', b'xbc'),
    (b' ', b'kabhbhkabh')
]

SPECIAL_TOKENS = ["<|endoftext|>", "<|endofthing|>"]
EXPECTED_VOCAB = {i: bytes([i]) for i in range(256)}
for tok in SPECIAL_TOKENS:
    token_utf8 = tok.encode("utf-8")
    EXPECTED_VOCAB[len(EXPECTED_VOCAB)] = token_utf8

for b1, b2 in EXPECTED_MERGES:
    EXPECTED_VOCAB[len(EXPECTED_VOCAB)] = b1+b2

EXPECTED_STATES = [
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'k', b'a'): 10,
                (b'a', b'b'): 16,
                (b'b', b'h'): 11,
                (b'h', b'b'): 1,
                (b'h', b'k'): 5,
                (b' ', b'b'): 3,
                (b'b', b'x'): 6,
                (b'x', b'z'): 3,
                (b' ', b'k'): 4,
                (b' ', b'a'): 6,
                (b'b', b'c'): 5,
                (b'x', b'y'): 3,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b'k', b'a', b'b', b'h', b'k', b'a', b'b', b'h'):  Counter({
                    (b'k', b'a') : 2,
                    (b'a', b'b'): 2,
                    (b'b', b'h'): 2,
                    (b'h', b'k'): 1,
            }),
            (b' ', b'k', b'a', b'b', b'h', b'b', b'h', b'k', b'a', b'b', b'h'):  Counter({
                    (b' ', b'k'): 1,
                    (b'k', b'a'): 2,
                    (b'a', b'b'): 2,
                    (b'b', b'h'): 3,
                    (b'h', b'k'): 1,
                    (b'h', b'b'): 1,
            }),
            (b' ', b'k', b'a', b'b', b'h', b'k', b'a', b'b', b'h'):  Counter({
                    (b'k', b'a') : 2,
                    (b'a', b'b'): 2,
                    (b'b', b'h'): 2,
                    (b'h', b'k'): 1,
                    (b' ', b'k'): 1,
            }),
            (b' ', b'b', b'x', b'z'): Counter({
                (b' ', b'b'): 1,
                (b'b', b'x'): 1,
                (b'x', b'z'): 1,
            }),
            (b' ', b'a', b'b', b'c'): Counter({
                (b' ', b'a'): 1,
                (b'a', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'a', b'b', b'x', b'y'): Counter({
                (b' ', b'a'): 1,
                (b'a', b'b'): 1,
                (b'b', b'x'): 1,
                (b'x', b'y'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b'k', b'a', b'b', b'h', b'k', b'a', b'b', b'h'): 1,
            (b' ', b'b', b'x', b'z'): 3,
            (b' ', b'k', b'a', b'b', b'h', b'k', b'a', b'b', b'h'): 3,
            (b' ', b'k', b'a', b'b', b'h', b'b', b'h', b'k', b'a', b'b', b'h'): 1,
            (b' ', b'a', b'b', b'c'): 3,
            (b' ', b'a', b'b', b'x', b'y'): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'k', b'ab'): 10,
                (b'ab', b'h'): 10,
                (b'b', b'h'): 1,
                (b'h', b'b'): 1,
                (b'h', b'k'): 5,
                (b' ', b'b'): 3,
                (b'ab', b'x'): 3,
                (b'b', b'x'): 3,
                (b'x', b'z'): 3,
                (b' ', b'k'): 4,
                (b' ', b'ab'): 6,
                (b'ab', b'c'): 3,
                (b'b', b'c'): 2,
                (b'x', b'y'): 3,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b'k', b'ab', b'h', b'k', b'ab', b'h'):  Counter({
                    (b'k', b'ab') : 2,
                    (b'ab', b'h'): 2,
                    (b'h', b'k'): 1,
            }),
            (b' ', b'k', b'ab', b'h', b'k', b'ab', b'h'):  Counter({
                    (b'k', b'ab') : 2,
                    (b'ab', b'h'): 2,
                    (b'h', b'k'): 1,
                    (b' ', b'k'): 1,
            }),
            (b' ', b'k', b'ab', b'h', b'b', b'h', b'k', b'ab', b'h'):  Counter({
                    (b'k', b'ab') : 2,
                    (b'ab', b'h'): 2,
                    (b'b', b'h'): 1,
                    (b'h', b'k'): 1,
                    (b' ', b'k'): 1,
                    (b'h', b'b'): 1,
            }),
            (b' ', b'b', b'x', b'z'): Counter({
                (b' ', b'b'): 1,
                (b'b', b'x'): 1,
                (b'x', b'z'): 1,
            }),
            (b' ', b'ab', b'c'): Counter({
                (b' ', b'ab'): 1,
                (b'ab', b'c'): 1,
            }),
            (b' ', b'ab', b'x', b'y'): Counter({
                (b' ', b'ab'): 1,
                (b'ab', b'x'): 1,
                (b'x', b'y'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b'k', b'ab', b'h', b'k', b'ab', b'h'): 1,
            (b' ', b'b', b'x', b'z'): 3,
            (b' ', b'k', b'ab', b'h', b'k', b'ab', b'h'): 3,
            (b' ', b'k', b'ab', b'h', b'b', b'h', b'k', b'ab', b'h'): 1,
            (b' ', b'ab', b'c'): 3,
            (b' ', b'ab', b'x', b'y'): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'kab', b'h'): 10,
                (b'b', b'h'): 1,
                (b'h', b'b'): 1,
                (b'h', b'kab'): 5,
                (b' ', b'b'): 3,
                (b'ab', b'x'): 3,
                (b'b', b'x'): 3,
                (b'x', b'z'): 3,
                (b' ', b'kab'): 4,
                (b' ', b'ab'): 6,
                (b'ab', b'c'): 3,
                (b'b', b'c'): 2,
                (b'x', b'y'): 3,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b'kab', b'h', b'kab', b'h'):  Counter({
                    (b'kab', b'h'): 2,
                    (b'h', b'kab'): 1,
            }),
            (b' ', b'kab', b'h', b'kab', b'h'):  Counter({
                    (b'kab', b'h'): 2,
                    (b'h', b'kab'): 1,
                    (b' ', b'kab'): 1,
            }),
            (b' ', b'kab', b'h', b'b', b'h', b'kab', b'h'):  Counter({
                    (b'kab', b'h'): 2,
                    (b'b', b'h'): 1,
                    (b'h', b'kab'): 1,
                    (b' ', b'kab'): 1,
                    (b'h', b'b'): 1,
            }),
            (b' ', b'b', b'x', b'z'): Counter({
                (b' ', b'b'): 1,
                (b'b', b'x'): 1,
                (b'x', b'z'): 1,
            }),
            (b' ', b'ab', b'c'): Counter({
                (b' ', b'ab'): 1,
                (b'ab', b'c'): 1,
            }),
            (b' ', b'ab', b'x', b'y'): Counter({
                (b' ', b'ab'): 1,
                (b'ab', b'x'): 1,
                (b'x', b'y'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b'kab', b'h', b'kab', b'h'): 1,
            (b' ', b'b', b'x', b'z'): 3,
            (b' ', b'kab', b'h', b'kab', b'h'): 3,
            (b' ', b'kab', b'h', b'b', b'h', b'kab', b'h'): 1,
            (b' ', b'ab', b'c'): 3,
            (b' ', b'ab', b'x', b'y'): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b'kabh', b'kabh'): 4,
                (b'h', b'kabh'): 1,
                (b' ', b'b'): 3,
                (b'ab', b'x'): 3,
                (b'b', b'x'): 3,
                (b'x', b'z'): 3,
                (b' ', b'kabh'): 4,
                (b' ', b'ab'): 6,
                (b'ab', b'c'): 3,
                (b'b', b'c'): 2,
                (b'x', b'y'): 3,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b'kabh', b'kabh'):  Counter({
                    (b'kabh', b'kabh'): 1,
            }),
            (b' ', b'kabh', b'kabh'):  Counter({
                    (b'kabh', b'kabh'): 1,
                    (b' ', b'kabh'): 1,
            }),
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'b', b'x', b'z'): Counter({
                (b' ', b'b'): 1,
                (b'b', b'x'): 1,
                (b'x', b'z'): 1,
            }),
            (b' ', b'ab', b'c'): Counter({
                (b' ', b'ab'): 1,
                (b'ab', b'c'): 1,
            }),
            (b' ', b'ab', b'x', b'y'): Counter({
                (b' ', b'ab'): 1,
                (b'ab', b'x'): 1,
                (b'x', b'y'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b'kabh', b'kabh'): 1,
            (b' ', b'b', b'x', b'z'): 3,
            (b' ', b'kabh', b'kabh'): 3,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' ', b'ab', b'c'): 3,
            (b' ', b'ab', b'x', b'y'): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b'kabh', b'kabh'): 4,
                (b'h', b'kabh'): 1,
                (b' ', b'b'): 3,
                (b' ab', b'x'): 3,
                (b'b', b'x'): 3,
                (b'x', b'z'): 3,
                (b' ', b'kabh'): 4,
                (b' ab', b'c'): 3,
                (b'b', b'c'): 2,
                (b'x', b'y'): 3,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b'kabh', b'kabh'):  Counter({
                    (b'kabh', b'kabh'): 1,
            }),
            (b' ', b'kabh', b'kabh'):  Counter({
                    (b'kabh', b'kabh'): 1,
                    (b' ', b'kabh'): 1,
            }),
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'b', b'x', b'z'): Counter({
                (b' ', b'b'): 1,
                (b'b', b'x'): 1,
                (b'x', b'z'): 1,
            }),
            (b' ab', b'c'): Counter({
                (b' ab', b'c'): 1,
            }),
            (b' ab', b'x', b'y'): Counter({
                (b' ab', b'x'): 1,
                (b'x', b'y'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b'kabh', b'kabh'): 1,
            (b' ', b'b', b'x', b'z'): 3,
            (b' ', b'kabh', b'kabh'): 3,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' ab', b'c'): 3,
            (b' ab', b'x', b'y'): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b' ', b'kabhkabh'): 3,
                (b'h', b'kabh'): 1,
                (b' ', b'b'): 3,
                (b' ab', b'x'): 3,
                (b'b', b'x'): 3,
                (b'x', b'z'): 3,
                (b' ', b'kabh'): 1,
                (b' ab', b'c'): 3,
                (b'b', b'c'): 2,
                (b'x', b'y'): 3,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhkabh'):  Counter({
                    (b' ', b'kabhkabh'): 1,
            }),
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'b', b'x', b'z'): Counter({
                (b' ', b'b'): 1,
                (b'b', b'x'): 1,
                (b'x', b'z'): 1,
            }),
            (b' ab', b'c'): Counter({
                (b' ab', b'c'): 1,
            }),
            (b' ab', b'x', b'y'): Counter({
                (b' ab', b'x'): 1,
                (b'x', b'y'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' ', b'b', b'x', b'z'): 3,
            (b' ', b'kabhkabh'): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' ab', b'c'): 3,
            (b' ab', b'x', b'y'): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b' ', b'kabhkabh'): 3,
                (b'h', b'kabh'): 1,
                (b' ', b'b'): 3,
                (b' ab', b'x'): 3,
                (b'b', b'xz'): 3,
                (b' ', b'kabh'): 1,
                (b' ab', b'c'): 3,
                (b'b', b'c'): 2,
                (b'x', b'y'): 3,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhkabh'):  Counter({
                    (b' ', b'kabhkabh'): 1,
            }),
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'b', b'xz'): Counter({
                (b' ', b'b'): 1,
                (b'b', b'xz'): 1,
            }),
            (b' ab', b'c'): Counter({
                (b' ab', b'c'): 1,
            }),
            (b' ab', b'x', b'y'): Counter({
                (b' ab', b'x'): 1,
                (b'x', b'y'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' ', b'b', b'xz'): 3,
            (b' ', b'kabhkabh'): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' ab', b'c'): 3,
            (b' ab', b'x', b'y'): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b' ', b'kabhkabh'): 3,
                (b'h', b'kabh'): 1,
                (b' ', b'b'): 3,
                (b' ab', b'xy'): 3,
                (b'b', b'xz'): 3,
                (b' ', b'kabh'): 1,
                (b' ab', b'c'): 3,
                (b'b', b'c'): 2,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhkabh'):  Counter({
                    (b' ', b'kabhkabh'): 1,
            }),
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'b', b'xz'): Counter({
                (b' ', b'b'): 1,
                (b'b', b'xz'): 1,
            }),
            (b' ab', b'c'): Counter({
                (b' ab', b'c'): 1,
            }),
            (b' ab', b'xy'): Counter({
                (b' ab', b'xy'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' ', b'b', b'xz'): 3,
            (b' ', b'kabhkabh'): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' ab', b'c'): 3,
            (b' ab', b'xy'): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b' ', b'kabhkabh'): 3,
                (b'h', b'kabh'): 1,
                (b' ', b'bxz'): 3,
                (b' ab', b'xy'): 3,
                (b' ', b'kabh'): 1,
                (b' ab', b'c'): 3,
                (b'b', b'c'): 2,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhkabh'):  Counter({
                    (b' ', b'kabhkabh'): 1,
            }),
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'bxz'): Counter({
                (b' ', b'bxz'): 1,
            }),
            (b' ab', b'c'): Counter({
                (b' ab', b'c'): 1,
            }),
            (b' ab', b'xy'): Counter({
                (b' ab', b'xy'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' ', b'bxz'): 3,
            (b' ', b'kabhkabh'): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' ab', b'c'): 3,
            (b' ab', b'xy'): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b' ', b'kabhkabh'): 3,
                (b'h', b'kabh'): 1,
                (b' ', b'bxz'): 3,
                (b' ', b'kabh'): 1,
                (b' ab', b'c'): 3,
                (b'b', b'c'): 2,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhkabh'):  Counter({
                    (b' ', b'kabhkabh'): 1,
            }),
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'bxz'): Counter({
                (b' ', b'bxz'): 1,
            }),
            (b' ab', b'c'): Counter({
                (b' ab', b'c'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' ', b'bxz'): 3,
            (b' ', b'kabhkabh'): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' ab', b'c'): 3,
            (b' abxy',): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b' ', b'kabhkabh'): 3,
                (b'h', b'kabh'): 1,
                (b' ', b'bxz'): 3,
                (b' ', b'kabh'): 1,
                (b'b', b'c'): 2,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhkabh'):  Counter({
                    (b' ', b'kabhkabh'): 1,
            }),
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'bxz'): Counter({
                (b' ', b'bxz'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' ', b'bxz'): 3,
            (b' ', b'kabhkabh'): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b'h', b'kabh'): 1,
                (b' ', b'bxz'): 3,
                (b' ', b'kabh'): 1,
                (b'b', b'c'): 2,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'bxz'): Counter({
                (b' ', b'bxz'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' ', b'bxz'): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'x'): 1,
                (b'x', b'b'): 2,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b'h', b'kabh'): 1,
                (b' ', b'kabh'): 1,
                (b'b', b'c'): 2,
                (b' ', b'x'): 2,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
            (b' ', b'x', b'x', b'b', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'x'): 1,
                (b'x', b'b'): 1,
                (b'b', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'x', b'b', b'c'): 1,
            (b' ', b'x', b'x', b'b', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'xb'): 1,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b'h', b'kabh'): 1,
                (b' ', b'kabh'): 1,
                (b'xb', b'c'): 2,
                (b' ', b'x'): 1,
                (b' ', b'xb'): 1,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'xb', b'c'): Counter({
                (b' ', b'xb'): 1,
                (b'xb', b'c'): 1,
            }),
            (b' ', b'x', b'xb', b'c'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'xb'): 1,
                (b'xb', b'c'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'xb', b'c'): 1,
            (b' ', b'x', b'xb', b'c'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'x', b'xbc'): 1,
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b'h', b'kabh'): 1,
                (b' ', b'kabh'): 1,
                (b' ', b'x'): 1,
                (b' ', b'xbc'): 1,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'xbc'): Counter({
                (b' ', b'xbc'): 1,
            }),
            (b' ', b'x', b'xbc'): Counter({
                (b' ', b'x'): 1,
                (b'x', b'xbc'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'xbc'): 1,
            (b' ', b'x', b'xbc'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'b', b'h'): 1,
                (b'kabh', b'b'): 1,
                (b'h', b'kabh'): 1,
                (b' ', b'kabh'): 1,
                (b' ', b'xxbc'): 1,
                (b' ', b'xbc'): 1,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabh', b'b', b'h', b'kabh'):  Counter({
                    (b'kabh', b'b'): 1,
                    (b' ', b'kabh'): 1,
                    (b'h', b'kabh'): 1,
                    (b'b', b'h'): 1,
            }),
            (b' ', b'xbc'): Counter({
                (b' ', b'xbc'): 1,
            }),
            (b' ', b'xxbc'): Counter({
                (b' ', b'xxbc'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabh', b'b', b'h', b'kabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'xbc'): 1,
            (b' ', b'xxbc'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'kabhb', b'h'): 1,
                (b'h', b'kabh'): 1,
                (b' ', b'kabhb'): 1,
                (b' ', b'xxbc'): 1,
                (b' ', b'xbc'): 1,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhb', b'h', b'kabh'):  Counter({
                    (b' ', b'kabhb'): 1,
                    (b'kabhb', b'h'): 1,
                    (b'h', b'kabh'): 1,
            }),
            (b' ', b'xbc'): Counter({
                (b' ', b'xbc'): 1,
            }),
            (b' ', b'xxbc'): Counter({
                (b' ', b'xxbc'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabhb', b'h', b'kabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'xbc'): 1,
            (b' ', b'xxbc'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b'kabhbh', b'kabh'): 1,
                (b' ', b'kabhbh'): 1,
                (b' ', b'xxbc'): 1,
                (b' ', b'xbc'): 1,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhbh', b'kabh'):  Counter({
                    (b' ', b'kabhbh'): 1,
                    (b'kabhbh', b'kabh'): 1,
            }),
            (b' ', b'xbc'): Counter({
                (b' ', b'xbc'): 1,
            }),
            (b' ', b'xxbc'): Counter({
                (b' ', b'xxbc'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabhbh', b'kabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'xbc'): 1,
            (b' ', b'xxbc'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b' ', b'kabhbhkabh'): 1,
                (b' ', b'xxbc'): 1,
                (b' ', b'xbc'): 1,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhbhkabh'):  Counter({
                    (b' ', b'kabhbhkabh'): 1,
            }),
            (b' ', b'xbc'): Counter({
                (b' ', b'xbc'): 1,
            }),
            (b' ', b'xxbc'): Counter({
                (b' ', b'xxbc'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabhbhkabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'xbc'): 1,
            (b' ', b'xxbc'): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b' ', b'kabhbhkabh'): 1,
                (b' ', b'xbc'): 1,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhbhkabh'):  Counter({
                    (b' ', b'kabhbhkabh'): 1,
            }),
            (b' ', b'xbc'): Counter({
                (b' ', b'xbc'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabhbhkabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' ', b'xbc'): 1,
            (b' xxbc',): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(
            {
                (b' ', b'kabhbhkabh'): 1,
            }
        ),
        "pairs_cache": _reverse_pair_index({
            (b' ', b'kabhbhkabh'):  Counter({
                    (b' ', b'kabhbhkabh'): 1,
            }),
        }),
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' ', b'kabhbhkabh'): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' xbc',): 1,
            (b' xxbc',): 1,
            (b' ',): 1,
        }
    },
    {
        "pairs_freqs": Counter(),
        "pairs_cache": {},
        "pretoken_freqs": {
            (b' bxz',): 3,
            (b' kabhkabh',): 3,
            (b'kabhkabh',): 1,
            (b' kabhbhkabh',): 1,
            (b' abc',): 3,
            (b' abxy',): 3,
            (b' xbc',): 1,
            (b' xxbc',): 1,
            (b' ',): 1,
        }
    },
]


class MockTestException(Exception):
    pass


def _mock_merge(_old_merge, throw_error_at, bpe_tokenizer: BPETokenizer):
    i = bpe_tokenizer._i

    before = EXPECTED_STATES[i]

    pairs_freqs_ct = Counter(
        {
            tup: freq for freq, tup, pres in bpe_tokenizer._pairs_freqs if pres
        }
    )
    assert pairs_freqs_ct == before["pairs_freqs"]
    assert bpe_tokenizer._pairs_cache == before["pairs_cache"]
    assert bpe_tokenizer._pretoken_freqs == before["pretoken_freqs"]

    ret = _old_merge()

    if i < len(EXPECTED_STATES)-1:
        after = EXPECTED_STATES[i+1]
        pairs_freqs_ct = Counter(
            {
                tup: freq for freq, tup, pres in bpe_tokenizer._pairs_freqs if pres
            }
        )

        assert pairs_freqs_ct == after["pairs_freqs"]
        assert bpe_tokenizer._pairs_cache == after["pairs_cache"]
        assert bpe_tokenizer._pretoken_freqs == after["pretoken_freqs"]
    else:
        assert not ret

    if throw_error_at and i == throw_error_at:
        raise MockTestException()

    return ret


def test_bpe_step_by_step():
    bpe_tokenizer = BPETokenizer(
        special_tokens=SPECIAL_TOKENS, num_processes=10
    )
    _old_merge = bpe_tokenizer._merge
    bpe_tokenizer._merge = partial(
        _mock_merge, _old_merge, False, bpe_tokenizer
    )

    bpe_tokenizer.train("cs336_basics/tokenizer/tests/data/owt_debug.txt")

    assert bpe_tokenizer.merges == EXPECTED_MERGES
    assert bpe_tokenizer.dictionary == EXPECTED_VOCAB


def test_bpe_load_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        bpe_tokenizer = BPETokenizer(
            special_tokens=SPECIAL_TOKENS,
            num_processes=10,
            checkpoint_dir=tmp,
        )

        _old_merge = bpe_tokenizer._merge
        save_every = 5
        error_step = 7
        bpe_tokenizer._merge = partial(
            _mock_merge, _old_merge, error_step, bpe_tokenizer
        )

        try:
            bpe_tokenizer.train(
                "cs336_basics/tokenizer/tests/data/owt_debug.txt",
                save_every=5
            )
        except MockTestException:
            print("Mock failure happened")

        bpe_tokenizer = BPETokenizer(
            special_tokens=SPECIAL_TOKENS,
            num_processes=10,
            checkpoint_dir=tmp,
        )
        _old_merge = bpe_tokenizer._merge
        bpe_tokenizer._merge = partial(
            _mock_merge, _old_merge, None, bpe_tokenizer
        )
        bpe_tokenizer.load_checkpoint()
        bpe_tokenizer.train(
                "cs336_basics/tokenizer/tests/data/owt_debug.txt",
                save_every=save_every
            )

        assert bpe_tokenizer.merges == EXPECTED_MERGES
        assert bpe_tokenizer.dictionary == EXPECTED_VOCAB
