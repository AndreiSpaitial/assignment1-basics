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

        self._token_pairs_cache = {}

    def _str_to_bytes_tuple(self, input_str: str) -> tuple[bytes]:
        input_str_utf8 = input_str.encode("utf-8")
        return tuple(bytes([el]) for el in input_str_utf8)

    def _token_pairs(self, pretoken: tuple[bytes]) -> Counter[tuple[bytes], int]:
        if pretoken in self._token_pairs_cache:
            return self._token_pairs_cache[pretoken]
        ret = Counter((p1, p2) for p1,p2 in pairwise(pretoken))

        self._token_pairs_cache[pretoken] = ret
        return ret

    def _pretokenize_chunk(self, chunk: str) -> tuple[
            Counter[tuple[bytes], int], 
            dict[tuple[bytes], Counter[tuple[bytes], int]]
        ]:
        split_regex = "|".join(re.escape(el) for el in self.special_tokens)
        token_freq = Counter()
        pairs_cache: dict[tuple[bytes], Counter[tuple[bytes], int]] = defaultdict(Counter)

        for doc in re.split(split_regex, chunk):
            for pre_token in re.finditer(PRETOKENIZER_PAT, doc):
                pretoken = self._str_to_bytes_tuple(pre_token.group())
                pairs = self._token_pairs(pretoken)

                for pair, freq in pairs.items():
                    pairs_cache[pair][pretoken] = freq
                
                token_freq[pretoken] += 1

        return token_freq, pairs_cache
    
    def _pretokenize(self, file_name) -> tuple[
            Counter[tuple[bytes], int], 
            dict[tuple[bytes], Counter[tuple[bytes], int]]
        ]:
        with open(file_name, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")

            pretoken_freqs = Counter()
            pairs_cache = defaultdict(Counter)
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token

                pretoken_freqs_chunk, pairs_cache_chunk = self._pretokenize_chunk(chunk)
                pretoken_freqs += pretoken_freqs_chunk

                for pair, pretokens in pairs_cache_chunk.items():
                    pairs = pairs_cache[pair]
                    for pretoken, freq in pretokens.items():
                        if pretoken in pairs:
                            continue
                        pairs[pretoken] = freq

        return pretoken_freqs, pairs_cache

    def _merge_pretoken(self, pretoken: tuple[bytes], pair: tuple[bytes]) -> tuple[bytes]:
        i=0
        pairs = self._token_pairs_cache[pretoken]
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
        
        self._token_pairs_cache[pretoken] = pairs
        return pretoken, pairs

    def _merge(
            self, 
            pairs_freqs: Counter[tuple[bytes], int],
            pairs_cache: dict[tuple[bytes], Counter[tuple[bytes], int]],
            pretoken_freqs: Counter[tuple[bytes], int],
        ) -> tuple[
            dict[tuple[bytes], Counter[tuple[bytes], int]], 
            Counter[tuple[bytes], int]
        ]:

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
            return False

        self.merges.append(max_pair)
        self.dictionary.append(max_pair[0]+max_pair[1])

        print(f"Chose {max_pair}")
        pretokens = pairs_cache.pop(max_pair)
        for pretoken in pretokens:
            pretoken_freq = pretoken_freqs.pop(pretoken)
            pairs = self._token_pairs(pretoken)
            
            print(f"{pretoken=:}")
            for pair, freq in pairs.items():
                if pair != max_pair:
                    pairs_cache[pair].pop(pretoken)
                pairs_freqs[pair] -= pretoken_freq*freq

                print(f"pairs_freqs[{pair}]={pairs_freqs[pair]}")
                if pairs_freqs[pair] < 0:
                    print(f"Failed with {pretoken=:} {pair=:}, {max_pair=:}, {pretoken_freq=:}, {freq=:}")

            new_pretoken, new_pairs = self._merge_pretoken(pretoken, max_pair)

            pretoken_freqs[new_pretoken] = pretoken_freq
            
            for pair,freq in new_pairs.items():
                pairs_cache[pair][new_pretoken] = freq
            
            merged_pair = max_pair[0] + max_pair[1]
            for new_pair, freq in new_pairs.items():
                new_pair_freq = freq * pretoken_freq
                pairs_freqs[new_pair] += new_pair_freq
                print(f"new pairs_freqs[{new_pair}]={pairs_freqs[new_pair]}")
        
        return True

    def train(self, file_name: str) -> None:
        start_time = time.time()
        pretoken_freqs, pairs_cache = self._pretokenize(file_name)
        end_time = time.time()

        print(f"Pre-tokenization done in {(end_time-start_time)}s")

        pairs_freqs: Counter[tuple[bytes], int] = Counter()
        for pair, pretokens in pairs_cache.items():
            for pretoken, pretoken_pair_freq in pretokens.items():
                freq = pretoken_freqs[pretoken]
                pairs_freqs[pair] += freq*pretoken_pair_freq

        i = 0
        found_pair = True
        start_time = time.time()
        with tqdm(total=self.vocab_size) as pbar:
            while found_pair and len(self.dictionary) < self.vocab_size:
                found_pair = self._merge(pairs_freqs, pairs_cache, pretoken_freqs)

                print(pairs_freqs)
                print("£££\n"*3)
                pbar.update(1)

        end_time = time.time()

        print(f"Merging done in {(end_time-start_time)}s")

    def tokenize(self, document: str) -> list[bytes]:
        ret = self._str_to_bytes_tuple(document)
        for merge in self.merges:
            ret = self._merge_pretoken(ret, merge)
        
        return ret
