import heapq
import multiprocessing
import os
import regex as re
import time
from collections import defaultdict, Counter
from itertools import pairwise
from typing import Iterable, Iterator

import torch
from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries


PRETOKENIZER_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(
            self,
            special_tokens: list[str] | None = None,
            vocab_size: int = 50_000,
            num_processes: int = 4,
            checkpoint_dir: str | os.PathLike = "",
            ):
        self.special_tokens = special_tokens
        self.dictionary: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.checkpoint_dir = checkpoint_dir

        i = len(self.dictionary)
        for el in special_tokens:
            token_utf8 = el.encode("utf-8")
            self.dictionary[i] = token_utf8
            i += 1

        self._dictionary_rev = {v: k for k, v in self.dictionary.items()}

        self.merges: list[tuple[bytes, bytes]] = []

        self.num_processes = num_processes
        self.vocab_size = vocab_size

        self._pairs_freqs: list[tuple[tuple[bytes, bytes], int, bool]] = []
        self._pairs_freqs_lookup: dict[tuple[bytes, bytes], list] = {}
        self._pairs_cache: dict[
            tuple[bytes, bytes], Counter[tuple[bytes]]
        ] = {}
        self._pretoken_freqs: Counter[tuple[bytes]] = Counter()

        self._i = 0

    def _str_to_bytes_tuple(self, input_str: str) -> tuple[bytes, ...]:
        input_str_utf8 = input_str.encode("utf-8")
        return tuple(bytes([el]) for el in input_str_utf8)

    def _token_pairs(
        self, pretoken: tuple[bytes, ...]
    ) -> Counter[tuple[bytes, bytes]]:
        ret = Counter((p1, p2) for p1, p2 in pairwise(pretoken))
        return ret

    def _pretokenize_chunk(self, chunk: str) -> Iterator[str]:
        split_regex = r"\b\B"
        if self.special_tokens:
            split_regex = "|".join(re.escape(el) for el in self.special_tokens)

        for doc in re.split(split_regex, chunk):
            for pre_token in re.finditer(PRETOKENIZER_PAT, doc):
                yield pre_token.group()

    def _process_pretoken_chunks(self, chunk: str) -> tuple[
            Counter[tuple[bytes, ...]],
            dict[tuple[bytes, bytes], Counter[tuple[bytes, ...]]],
            ]:
        token_freq: Counter[tuple[bytes, ...]] = Counter()
        pairs_cache: dict[
            tuple[bytes, bytes], Counter[tuple[bytes, ...]]
        ] = defaultdict(Counter)

        for pretoken_str in self._pretokenize_chunk(chunk):
            pretoken = self._str_to_bytes_tuple(pretoken_str)
            pairs = self._token_pairs(pretoken)

            for pair, freq in pairs.items():
                pairs_cache[pair][pretoken] = freq

            token_freq[pretoken] += 1

        return token_freq, pairs_cache

    def _process_chunk(self, file_name, start, end):
        with open(file_name, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token

            return self._process_pretoken_chunks(chunk)

    def _pretokenize(self, file_name) -> None:
        if len(self._pretoken_freqs) > 0 and len(self._pairs_cache) > 0:
            return

        pretoken_freqs = Counter()
        pairs_cache = defaultdict(Counter)

        with open(file_name, "rb") as fb:
            boundaries = find_chunk_boundaries(fb, self.num_processes, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            # map blocks until all results are ready
            results = pool.starmap(
                self._process_chunk,
                [
                    (file_name, start, end)
                    for start, end in zip(boundaries[:-1], boundaries[1:])
                ]
            )

        for pretoken_freqs_chunk, pairs_cache_chunk in results:
            pretoken_freqs += pretoken_freqs_chunk
            for pair, pretokens in pairs_cache_chunk.items():
                pairs = pairs_cache[pair]
                for pretoken, freq in pretokens.items():
                    if pretoken in pairs:
                        continue
                    pairs[pretoken] = freq

        self._pretoken_freqs = pretoken_freqs
        self._pairs_cache = pairs_cache

    def _merge_pretoken(
            self, pretoken: tuple[bytes, ...], pair: tuple[bytes, bytes]
            ) -> tuple[tuple[bytes, ...], Counter[tuple[bytes, bytes]]]:
        i = 0
        pairs = self._token_pairs(pretoken)

        merged_pair = pair[0] + pair[1]
        while i < len(pretoken)-1:
            if pretoken[i:(i+2)] == pair:
                pairs[pair] -= 1
                if pairs[pair] == 0:
                    pairs.pop(pair)
                if i-1 >= 0:
                    pairs[(pretoken[i-1], merged_pair)] += 1
                    new_pair = (pretoken[i-1], pair[0])
                    pairs[new_pair] -= 1
                    if pairs[new_pair] == 0:
                        pairs.pop(new_pair)
                if i+2 < len(pretoken):
                    pairs[(merged_pair, pretoken[i+2])] += 1
                    new_pair = (pair[1], pretoken[i+2])
                    pairs[new_pair] -= 1
                    if pairs[new_pair] == 0:
                        pairs.pop(new_pair)
                pretoken = pretoken[:i] + (merged_pair,) + pretoken[i+2:]
            else:
                i += 1

        return pretoken, pairs

    def _merge(self) -> bool:
        pairs_freqs = self._pairs_freqs
        pairs_freqs_lookup = self._pairs_freqs_lookup
        pairs_cache = self._pairs_cache
        pretoken_freqs = self._pretoken_freqs

        max_pair, max_freq = None, None
        while pairs_freqs:
            freq, pair, present = heapq._heappop_max(pairs_freqs)
            if not present:
                continue
            max_pair, max_freq = pair, freq
            break

        if max_pair is None:  # no more merges possible, all pre-tokens are a single bytes object
            return False

        self.merges.append(max_pair)
        self.dictionary[len(self.dictionary)] = max_pair[0]+max_pair[1]

        # print(f"Chose {max_pair}")
        pretokens = pairs_cache.pop(max_pair)
        for pretoken in pretokens:
            pretoken_freq = pretoken_freqs.pop(pretoken)
            pairs = self._token_pairs(pretoken)

            # print(f"{pretoken=:}")
            new_pretoken, new_pairs = self._merge_pretoken(pretoken, max_pair)
            pretoken_freqs[new_pretoken] = pretoken_freq
            # TODO: heapify this with heap replace
            for pair,freq in pairs.items():
                # print(f"pairs cache {pair}", pairs_cache[pair])
                if pair != max_pair:
                    pairs_cache[pair].pop(pretoken)
                    if len(pairs_cache[pair]) == 0:
                        pairs_cache.pop(pair)
                new_freq = new_pairs[pair]
                diff = (new_freq-freq)*pretoken_freq
                if diff == 0:
                    continue

                entry = pairs_freqs_lookup[pair]
                entry[-1] = False

                if entry[0]+diff != 0:
                    new_entry = [entry[0]+diff, entry[1], True]
                    pairs_freqs.append(new_entry)
                    heapq._siftdown_max(pairs_freqs, 0, len(pairs_freqs)-1)
                    pairs_freqs_lookup[pair] = new_entry
                else:
                    pairs_freqs_lookup.pop(pair)
                # print(f"pairs_freqs[{pair}]={pairs_freqs[pair]}")
            
            merged_pair = max_pair[0] + max_pair[1]
            for new_pair, freq in new_pairs.items():
                pairs_cache[new_pair][new_pretoken] = freq
                if new_pair in pairs:
                    continue
                new_pair_freq = freq * pretoken_freq
                
                entry = pairs_freqs_lookup.get(new_pair, [0, new_pair, True])
                entry[-1] = False
                new_entry = [entry[0]+new_pair_freq, new_pair, True]
                pairs_freqs_lookup[new_pair] = new_entry
                pairs_freqs.append(new_entry)
                heapq._siftdown_max(pairs_freqs, 0, len(pairs_freqs)-1)

                # print(f"new pairs_freqs[{new_pair}]={pairs_freqs[new_pair]}")
        
        # print(pairs_freqs)
        # print("£££\n"*3)

        return True

    def train(
        self, file_name: str | os.PathLike, save_every: int = 500
    ) -> None:
        if self._i == 0:
            start_time = time.time()
            self._pretokenize(file_name)
            end_time = time.time()

            # print(f"Pre-tokenization done in {(end_time-start_time)}s")

            pairs_freqs: list[
                tuple[tuple[bytes, bytes], int, bool]
            ] = self._pairs_freqs
            pairs_freq_ct: Counter[tuple[bytes, bytes]] = Counter()
            for pair, pretokens in self._pairs_cache.items():
                for pretoken, pretoken_pair_freq in pretokens.items():
                    freq = self._pretoken_freqs[pretoken]
                    pairs_freq_ct[pair] += freq*pretoken_pair_freq

            for tup, freq in pairs_freq_ct.items():
                entry = [freq, tup, True]
                pairs_freqs.append(entry)
            heapq._heapify_max(pairs_freqs)

            self._pairs_freqs = pairs_freqs

        pairs_freqs_lookup: dict[tuple[bytes, bytes], list[object]] = {}
        for entry in self._pairs_freqs:
            _, tup, pres = entry
            if pres:
                pairs_freqs_lookup[tup] = entry
        self._pairs_freqs_lookup = pairs_freqs_lookup

        found_pair = True
        start_time = time.time()
        with tqdm(initial=self._i, total=self.vocab_size) as pbar:
            while found_pair and len(self.dictionary) < self.vocab_size:
                found_pair = self._merge()

                # print(pairs_freqs)
                # print("£££\n"*3)
                pbar.update(1)
                self._i += 1

                if self._i % save_every == 0:
                    self.save_checkpoint()

        end_time = time.time()

        print(f"Merging done in {(end_time-start_time)}s")

    def state_dict(self, full=True) -> dict:
        if full:
            ret = {
                "special_tokens": self.special_tokens,
                "dictionary": self.dictionary,
                "merges": self.merges,
                "pairs_freqs": self._pairs_freqs,
                "pairs_cache": self._pairs_cache,
                "pretoken_freqs": self._pretoken_freqs,
                "i": self._i,
            }
        else:
            ret = {
                "special_tokens": self.special_tokens,
                "dictionary": self.dictionary,
                "merges": self.merges,
            }

        return ret

    def load_state_dict(self, new_dict) -> None:
        if "special_tokens" in new_dict:
            self.special_tokens = new_dict["special_tokens"]

        self.dictionary = new_dict["dictionary"]
        self.merges = new_dict["merges"]
        self._dictionary_rev = {v:k for k,v in self.dictionary.items()}

        # load training state if present
        if "pairs_freqs" in new_dict:
            self._pairs_freqs = new_dict["pairs_freqs"]
            self._pairs_cache = new_dict["pairs_cache"]
            self._pretoken_freqs = new_dict["pretoken_freqs"]
            self._i = new_dict["i"]

    def save_checkpoint(self) -> None:
        state_dict = self.state_dict()
        checkpoint_dir = os.path.join(self.checkpoint_dir, "bpe_cp.pth")
        torch.save(state_dict, checkpoint_dir)

    def load_checkpoint(self) -> None:
        checkpoint_dir = os.path.join(self.checkpoint_dir, "bpe_cp.pth")
        with torch.serialization.safe_globals([defaultdict, Counter]):
            state_dict = torch.load(checkpoint_dir)
        self.load_state_dict(state_dict)

    @staticmethod
    def from_files(
        cls, vocab_filepath, merges_filepath, special_tokens=None
    ) -> "BPETokenizer":
        vocab = torch.load(vocab_filepath)
        merges = torch.load(merges_filepath)

        ret = cls(special_tokens=special_tokens)

        ret.load_state_dict({
            "dictionary": vocab,
            "merges": merges,
        })

        return ret

    def _encode_pretoken(self, token: str) -> tuple[bytes, ...]:
        ret = self._str_to_bytes_tuple(token)
        for merge in self.merges:
            ret, _ = self._merge_pretoken(ret, merge)

        return ret

    def encode(self, text: str) -> list[int]:
        ret = []
        for pretoken in self._pretokenize_chunk(text):
            encoded_pretoken = self._encode_pretoken(pretoken)
            for word in encoded_pretoken:
                ret.append(self._dictionary_rev[word])

        return ret

    def encode_iterable(self, text_stream: Iterable[str]) -> Iterator[int]:
        for text in text_stream:
            for pretoken in self._pretokenize_chunk(text):
                encoded_pretoken = self._encode_pretoken(pretoken)
                for word in encoded_pretoken:
                    yield self._dictionary_rev[word]

    def decode(self, ids: list[int]) -> str:
        ret_l = []
        for el in ids:
            token = self.dictionary[el]
            ret_l.append(token)

        ret = b"".join(ret_l)

        return ret.decode("utf-8", errors="replace")
