from concurrent.futures import Future, ProcessPoolExecutor
import time
from pathlib import Path
import regex as re
import os
from typing import BinaryIO

Tokens = tuple[bytes, ...]
BytePair = tuple[bytes, bytes]

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
        self.pretoken_pattern: re.Pattern[str] = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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
        num_processes = 8
        with open(self.file, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, self.special_tokens[0].encode("utf-8"))
        futures: list[Future[dict[str, int]]] = []
        with ProcessPoolExecutor() as executor:
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

    def generate_byte_pairs(self, pretokens: list[Tokens], counts: list[int]) -> dict[BytePair, int]:
        byte_pairs: dict[BytePair, int] = {}
        for token, freq in zip(pretokens, counts):
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                byte_pairs[pair] = byte_pairs.get(pair, 0) + freq
        return byte_pairs

    def merge(
        self,
        byte_pairs: dict[BytePair, int],
        pretokens: list[Tokens],
        freqs: list[int],
        pair: BytePair,
        token_id: int,
    ) -> None:
        pair_as_bytes = pair[0] + pair[1]
        self.vocab[token_id] = pair_as_bytes
        self.merges.append(pair)

        for i in range(len(pretokens)):
            freq = freqs[i]
            j = 0
            while j < len(pretokens[i]) - 1:
                pretoken = pretokens[i]
                if (pretoken[j], pretoken[j + 1]) == pair:
                    pretokens[i] = pretoken[:j] + (pair_as_bytes,) + pretoken[j + 2:]
                    if j > 0:
                        left = (pretoken[j - 1], pretoken[j])
                        new_left = (pretoken[j - 1], pair_as_bytes)
                        byte_pairs[left] -= freq
                        byte_pairs[new_left] = byte_pairs.get(new_left, 0) + freq
                    if j + 2 < len(pretoken):
                        right = (pretoken[j + 1], pretoken[j + 2])
                        new_right = (pair_as_bytes, pretoken[j + 2])
                        byte_pairs[right] -= freq
                        byte_pairs[new_right] = byte_pairs.get(new_right, 0) + freq
                j += 1
        del byte_pairs[pair]

    def train(self):
        self.populate_vocab()
        pretokens, count = self.get_pretokenized_corpus()
        byte_pairs = self.generate_byte_pairs(pretokens, count)
        arg_max = max(((freq, pair) for pair, freq in byte_pairs.items()), default=None)
        vocab_len = len(self.vocab)
        for i in range(self.extended_vocab_count):
            if not arg_max:
                print("no arg max!")
                break
            if arg_max[0] == 0:
                break
            self.merge(byte_pairs, pretokens, count, arg_max[1], vocab_len + i)
            # can be optimized to be dynamic
            arg_max = max(((freq, pair) for pair, freq in byte_pairs.items()), default=None)
        # self.dump("final", pretoken_set, byte_pairs)

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
    text = Path("../data/text.txt")
    bpe = BPETrainer(text, 1000, ["<|endoftext|>"])
    bpe.train()
    print(bpe.get_merges())
    print(bpe.get_vocab())

