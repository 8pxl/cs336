from concurrent.futures import Future, ProcessPoolExecutor
import heapq
import time
from pathlib import Path
import regex as re
import os
from typing import BinaryIO

from sympy import Max

Tokens = tuple[bytes, ...]
BytePair = tuple[bytes, bytes]

class MaxPair:
    __slots__ = ("pair",)
    def __init__(self, pair: BytePair) -> None:
        self.pair = pair
    def __lt__(self, other: "MaxPair") -> bool:
        return self.pair > other.pair

def show(tok: bytes) -> str:
    return tok.decode("utf-8", errors="backslashreplace")


def show_pretoken(pretoken: tuple[bytes, ...]) -> str:
    # '|' marks the token boundaries, so merges are visible: l|o|w -> lo|w
    return "|".join(show(t) for t in pretoken)


def show_pair(pair: BytePair) -> str:
    return f"{show(pair[0])}+{show(pair[1])}"

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    _ = file.seek(0, os.SEEK_END)
    file_size = file.tell()
    _ = file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        _ = file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

class BPETrainer:
    def __init__(self, file: Path, max_vocab_size: int, special_tokens: list[str]) -> None:
        self.extended_vocab_count: int = max_vocab_size - 256 - len(special_tokens) # 256 bytes - one special char
        # self.corpus: str = corpus
        self.special_tokens: list[str] = special_tokens
        self.file: Path = file
        self.vocab: dict[int, bytes] = {}
        self.merges: list[BytePair] = []
        self.timings: dict[str, float] = {}
        self.merge_times: list[float] = []
        self.n_merges: int = 0
        self.n_pretokens: int = 0
        self.pretoken_pattern: re.Pattern[str] = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.pair_heap: list[tuple[int, MaxPair]] = []

    def get_merges(self):
        return self.merges
    def get_vocab(self):
        return self.vocab
    def populate_vocab(self):
        for i in range(256):
            self.vocab[i] = i.to_bytes()
        count = 256
        for token in self.special_tokens:
            self.vocab[count] = token.encode("utf-8")
            count += 1

    def get_pretokenized_corpus(self) -> tuple[list[Tokens], list[int]]:
        #each pre token is a tuple of bytes
        num_processes = 12
        with open(self.file, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, self.special_tokens[0].encode("utf-8"))
        futures: list[Future[dict[str, int]]] = []
        with ProcessPoolExecutor(max_workers=12) as executor:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                future = executor.submit(self.pretokenize_chunk, start, end)
                futures.append(future)

        while all([future.running() for future in futures]):
            time.sleep(0.5)

        acc: dict[str, int] = {}
        for f in futures:
            for s, count in f.result().items():
                acc[s] = acc.get(s, 0) + count

        pretokens: list[Tokens] = []
        freqs: list[int] = []
        for s, n in acc.items():
            pretokens.append(tuple(b.to_bytes() for b in s.encode("utf-8")))
            freqs.append(n)
        return pretokens, freqs

    def pretokenize_chunk(self, start: int, end: int) -> dict[str,int]:
        with open(self.file, "rb") as f:
            _ = f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            token_pattern = "|".join(re.escape(t) for t in sorted(self.special_tokens, key = len, reverse=True))
            split_chunk = re.split(token_pattern, chunk)
            pretoken_freq: dict[str, int] = {}
            for segment in split_chunk:
                for m in re.finditer(self.pretoken_pattern, segment):
                    pretoken_freq[m[0]] = 1 + pretoken_freq.get(m[0], 0)
        return pretoken_freq

    def generate_byte_pairs(self, pretokens: list[Tokens], counts: list[int]) -> tuple[dict[BytePair, int], dict[BytePair, set[int]]]:
        byte_pairs: dict[BytePair, int] = {}
        pair_locations: dict[BytePair, set[int]] = {}
        token_index = 0
        for token, freq in zip(pretokens, counts):
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                count = byte_pairs.get(pair, 0) + freq
                byte_pairs[pair] = count
                heapq.heappush(self.pair_heap, (-count, MaxPair(pair)))
                if pair not in pair_locations:
                    pair_locations[pair] = {token_index}
                else:
                    pair_locations[pair].add(token_index)
            token_index += 1
        return byte_pairs, pair_locations

    def merge(
        self,
        byte_pairs: dict[BytePair, int],
        pretokens: list[Tokens],
        freqs: list[int],
        pair: BytePair,
        pair_map: dict[BytePair, set[int]],
        token_id: int,
    ) -> None:
        pair_as_bytes = pair[0] + pair[1]
        self.vocab[token_id] = pair_as_bytes
        self.merges.append(pair)

        for i in pair_map[pair]:
            freq = freqs[i]
            j = 0
            while j < len(pretokens[i]) - 1:
                pretoken = pretokens[i]
                if (pretoken[j], pretoken[j + 1]) == pair:
                    pretokens[i] = pretoken[:j] + (pair_as_bytes,) + pretoken[j + 2:]
                    if j > 0:
                        left: BytePair = (pretoken[j - 1], pretoken[j])
                        new_left = (pretoken[j - 1], pair_as_bytes)
                        #intentionally not clearing the right / left pairs because it cold appear later in the same token and i dont want to deal with ts
                        left_freq = byte_pairs[left] - freq
                        byte_pairs[left] = left_freq

                        new_left_freq = byte_pairs.get(new_left, 0) + freq
                        byte_pairs[new_left] = new_left_freq

                        heapq.heappush(self.pair_heap, (-left_freq, MaxPair(left)))
                        heapq.heappush(self.pair_heap, (-new_left_freq, MaxPair(new_left)))

                        if new_left not in pair_map:
                            pair_map[new_left] = {i}
                        else:
                            pair_map[new_left].add(i)

                    if j + 2 < len(pretoken):
                        right = (pretoken[j + 1], pretoken[j + 2])
                        new_right = (pair_as_bytes, pretoken[j + 2])
                        right_freq = byte_pairs[right] - freq

                        byte_pairs[right] = right_freq
                        
                        #intentionally not clearing the right / left pairs because it cold appear later in the same token and i dont want to deal with ts
                        new_right_freq = byte_pairs.get(new_right, 0) + freq
                        byte_pairs[new_right] = new_right_freq

                        heapq.heappush(self.pair_heap, (-right_freq, MaxPair(right)))
                        heapq.heappush(self.pair_heap, (-new_right_freq, MaxPair(new_right)))
                        if new_right not in pair_map:
                            pair_map[new_right] = {i}
                        else:
                            pair_map[new_right].add(i)
                j += 1
        del byte_pairs[pair]
        del pair_map[pair]

    def train(self, verbose: bool = False) -> None:
        t = time.perf_counter()
        self.populate_vocab()
        pretokens, count = self.get_pretokenized_corpus()
        t_pretokenize = time.perf_counter() - t

        t = time.perf_counter()
        byte_pairs, pair_map = self.generate_byte_pairs(pretokens, count)
        t_initial_count = time.perf_counter() - t

        t_argmax = 0.0
        t_merge = 0.0
        n_merges = 0
        vocab_len = len(self.vocab)

        t = time.perf_counter()
        # arg_max = max(((freq, pair) for pair, freq in byte_pairs.items()), default=None)
        arg_max = heapq.heappop(self.pair_heap)
        t_argmax += time.perf_counter() - t

        for i in range(self.extended_vocab_count):
            print(f"progress: {(100.0 * (i / self.extended_vocab_count)):.3f}%")
            if not arg_max:
                print("no arg max!")
                break
            if arg_max[0] == 0:
                break

            t = time.perf_counter()
            self.merge(byte_pairs, pretokens, count, arg_max[1].pair, pair_map, vocab_len + i)
            dt = time.perf_counter() - t
            t_merge += dt
            self.merge_times.append(dt)
            n_merges += 1

            # can be optimized to be dynamic
            t = time.perf_counter()
            # arg_max = max(((freq, pair) for pair, freq in byte_pairs.items()), default=None)
            arg_max = heapq.heappop(self.pair_heap)
            while byte_pairs[arg_max[1].pair] != arg_max[0]:
                arg_max = heapq.heappop(self.pair_heap)

            t_argmax += time.perf_counter() - t

        self.timings = {
            "pretokenize": t_pretokenize,
            "initial_pair_count": t_initial_count,
            "merge": t_merge,
            "argmax": t_argmax,
        }
        self.timings["total"] = sum(self.timings.values())
        self.n_merges = n_merges
        self.n_pretokens = len(pretokens)

        if verbose:
            self.report_timings()

    def report_timings(self) -> None:
        total = self.timings["total"]
        print(f"\n===== timings ({self.n_merges} merges, {self.n_pretokens} distinct pretokens) =====")
        for name, secs in sorted(self.timings.items(), key=lambda kv: -kv[1]):
            if name == "total":
                continue
            print(f"  {name:<20} {secs:7.3f}s  {100 * secs / total:5.1f}%")
        print(f"  {'total':<20} {total:7.3f}s")
        if self.n_merges:
            print(f"  per merge: {1000 * (self.timings['merge'] + self.timings['argmax']) / self.n_merges:.3f} ms")

    def dump(
        self,
        label: str,
        pretoken_set: list[tuple[Tokens, int]],
        byte_pairs: dict[BytePair, int],
        top: int = 10,
    ) -> None:
        print(f"\n===== {label} =====")
        print("pretokens (freq):")
        for pretoken, n in sorted(pretoken_set, key=lambda kv: -kv[1]):
            print(f"  {show_pretoken(pretoken):<24} {n}")
        print(f"byte pairs (top {top}):")
        for pair, n in sorted(byte_pairs.items(), key=lambda kv: -kv[1])[:top]:
            print(f"  {show_pair(pair):<12} {n}")
        print(f"vocab: {' | '.join(show(token) for token in self.vocab.values())}")
    # def merge(self):


#
# content = Path("input.txt").read_text(encoding="utf-8")

if __name__ == "__main__":
    text = Path("../data/owt_valid.txt")
    bpe = BPETrainer(text, 32000, ["<|endoftext|>"])
    bpe.train(True)
    merges = bpe.get_merges()
    vocab = bpe.get_vocab()
    from cs336_basics.serialization import save, load

    vocab_path, merges_path = save(vocab, merges, Path("out/owt"))
    # vocab, merges = load(vocab_path, merges_path)

