import regex as re
import time
from collections import defaultdict, Counter
from itertools import pairwise

from tqdm import tqdm

from cs336_basics.pretokenization_example import find_chunk_boundaries


PRETOKENIZER_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, 
            special_tokens: list[str],
            vocab_size: int = 50_000,
            num_processes: int = 4,
        ):
        self.special_tokens = special_tokens
        self.dictionary: list[bytes] = [bytes([i]) for i in range(256)]
        self.dictionary.extend([el.encode("utf-8") for el in special_tokens])

        self.merges: list[tuple[bytes]] = []

        self.num_processes = num_processes
        self.vocab_size = vocab_size

    def _str_to_bytes_tuple(self, input_str: str) -> tuple[bytes]:
        input_str_utf8 = input_str.encode("utf-8")
        return tuple(bytes([el]) for el in input_str_utf8)

    def _token_pairs(self, pretoken: tuple[bytes]) -> Counter[tuple[bytes], int]:
        return Counter((p1, p2) for p1,p2 in pairwise(pretoken))

    def _pretokenize_chunk(self, chunk: str) -> tuple[dict[tuple[bytes], int], dict[tuple[bytes], Counter[tuple[bytes], int]]]:
        split_regex = "|".join(re.escape(el) for el in self.special_tokens)
        ret = Counter()
        pairs_cache: dict[tuple[bytes], Counter[tuple[bytes], int]] = {}

        for doc in re.split(split_regex, chunk):
            for pre_token in re.finditer(PRETOKENIZER_PAT, doc):
                pretoken = self._str_to_bytes_tuple(pre_token.group())
                if pretoken not in pairs_cache:
                    pairs_cache[pretoken] = self._token_pairs(pretoken)
                
                ret[pretoken] += 1
        
        return ret, pairs_cache
    
    def _pretokenize(self, file_name) -> tuple[Counter[tuple[bytes], int], dict[tuple[bytes], Counter[tuple[bytes], int]]]:
        with open(file_name, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")

            pretoken_freqs = Counter()
            pairs_cache = {}
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token

                pretoken_freqs_chunk, pairs_cache_chunk = self._pretokenize_chunk(chunk)
                pretoken_freqs += pretoken_freqs_chunk

                for pretoken, pairs in pairs_cache_chunk.items():
                    if pretoken in pairs_cache:
                        continue
                    pairs_cache[pretoken] = pairs

        return pretoken_freqs, pairs_cache

    def _merge_pretoken(self, pretoken: tuple[bytes], pair: tuple[bytes]) -> tuple[bytes]:
        i=0
        while i < len(pretoken)-1:
            if pretoken[i:(i+2)] == pair:
                pretoken = pretoken[:i] + (pretoken[i] + pretoken[i+1],) + pretoken[i+2:]
            else:
                i += 1
        
        return pretoken

    def _merge(
            self, 
            pairs_freqs: Counter[tuple[bytes], int],
            pairs_cache: dict[tuple[bytes], Counter[tuple[bytes], int]],
            pretoken_freqs: Counter[tuple[bytes], int],
        ) -> tuple[dict[tuple[bytes], Counter[tuple[bytes], int]], Counter[tuple[bytes], int]]:

        max_pair, max_freq = None,None
        for pair,freq in pairs_freqs.items():
            if freq == 0:
                continue
            if max_pair is None:
                max_pair = pair
            if max_freq is None:
                max_freq = freq
            
            max_freq, max_pair = max((max_freq, max_pair), (freq, pair))
        
        if max_pair is None: # no more merges possible, all pre-tokens are a single bytes object
            return None, None

        self.merges.append(max_pair)
        self.dictionary.append(max_pair[0]+max_pair[1])

        new_pairs_cache = {}
        new_pretoken_freqs = {}

        for pretoken, pairs in pairs_cache.items():
            pretoken_freq = pretoken_freqs[pretoken]
            if max_pair not in pairs:
                new_pairs_cache[pretoken] = pairs
                new_pretoken_freqs[pretoken] = pretoken_freq
                continue

            new_pretoken = self._merge_pretoken(pretoken, max_pair)
            new_pairs = self._token_pairs(new_pretoken)
            new_pairs_cache[new_pretoken] = new_pairs

            for pair, freq in pairs.items():
                new_freq = new_pairs[pair]
                delta = new_freq - freq
                pairs_freqs[pair] += pretoken_freq*delta
            
            for new_pair, freq in new_pairs.items():
                if new_pair in pairs:
                    continue
                new_pair_freq = freq * pretoken_freq
                pairs_freqs[new_pair] += new_pair_freq
            
            new_pretoken_freqs[new_pretoken] = pretoken_freq

        return new_pairs_cache, new_pretoken_freqs

    def train(self, file_name: str) -> None:
        start_time = time.time()
        pretoken_freqs, pairs_cache = self._pretokenize(file_name)
        end_time = time.time()

        print(f"Pre-tokenization done in {(end_time-start_time)}s")


        pairs_freqs: Counter[tuple[bytes], int] = Counter()
        for pretoken, pairs in pairs_cache.items():
            freq = pretoken_freqs[pretoken]
            for pair, pretoken_pair_freq in pairs.items():
                pairs_freqs[pair] += freq*pretoken_pair_freq

        i = 0
        with tqdm(total=self.vocab_size) as pbar:
            while pairs_cache is not None and len(self.dictionary) < self.vocab_size:
                pairs_cache, pretoken_freqs = self._merge(pairs_freqs, pairs_cache, pretoken_freqs)
                pbar.update(1)

    def tokenize(self, document: str) -> list[bytes]:
        ret = self._str_to_bytes_tuple(document)
        for merge in self.merges:
            ret = self._merge_pretoken(ret, merge)
        
        return ret
