import regex as re
from collections import defaultdict, Counter
from itertools import pairwise

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

    def _pretokenize_chunk(self, chunk: str) -> dict[str, int]:
        split_regex = "|".join(re.escape(el) for el in self.special_tokens)
        ret = Counter()

        for doc in re.split(split_regex, chunk):
            for pre_token in re.finditer(PRETOKENIZER_PAT, doc):
                ret[self._str_to_bytes_tuple(pre_token.group())] += 1
        
        return ret
    
    def _pretokenize(self, file_name) -> Counter[tuple[bytes], int]:
        with open(file_name, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")

            total_freqs = Counter()
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # Run pre-tokenization on your chunk and store the counts for each pre-token

                counts = self._pretokenize_chunk(chunk)
                total_freqs += counts
            
        return total_freqs

    def _merge_pretoken(self, pretoken: tuple[bytes], pair: tuple[bytes]) -> tuple[bytes]:
        i=0
        while i < len(pretoken)-1:
            if pretoken[i:(i+2)] == pair:
                pretoken = pretoken[:i] + (pretoken[i] + pretoken[i+1],) + pretoken[i+2:]
            else:
                i += 1
        
        return pretoken

    def _merge(self, total_freqs: Counter[tuple[bytes], int]) -> Counter[tuple[bytes], int]:
        pair_freqs = defaultdict(lambda:0)
        for pretoken, freq in total_freqs.items():
            for b1,b2 in pairwise(pretoken):
                pair_freqs[tuple([b1,b2])] += freq
        
        max_pair, max_freq = None,None
        for pair,freq in pair_freqs.items():
            if max_pair is None:
                max_pair = pair
            if max_freq is None:
                max_freq = freq
            
            max_freq, max_pair = max((max_freq, max_pair), (freq, pair))
        
        if max_pair is None: # no more merges possible, all pre-tokens are a single bytes object
            return

        self.merges.append(max_pair)
        self.dictionary.append(max_pair[0]+max_pair[1])

        ret = Counter()
        for pretoken, freq in total_freqs.items():
            new_pretoken = self._merge_pretoken(pretoken, max_pair)
            ret[new_pretoken] = freq

        return ret

    def train(self, file_name: str) -> None:
        total_freqs = self._pretokenize(file_name)

        while total_freqs is not None and len(self.dictionary) < self.vocab_size:
            total_freqs = self._merge(total_freqs)

    def tokenize(self, document: str) -> list[bytes]:
        ret = self._str_to_bytes_tuple(document)
        for merge in self.merges:
            ret = self._merge_pretoken(ret, merge)
        
        return ret
