from pathlib import Path

Tokens = tuple[bytes, ...]
BytePair = tuple[bytes, bytes]

def show(tok: bytes) -> str:
    return tok.decode("utf-8", errors="backslashreplace")


def show_pretoken(pretoken: tuple[bytes, ...]) -> str:
    # '|' marks the token boundaries, so merges are visible: l|o|w -> lo|w
    return "|".join(show(t) for t in pretoken)


def show_pair(pair: BytePair) -> str:
    return f"{show(pair[0])}+{show(pair[1])}"


class Tokenizer:
    def __init__(self, input: str, max_vocab_size: int) -> None:
        self.extended_vocab: list[bytes] = []
        self.max_vocab_size: int = max_vocab_size
        self.input: str = input

        # self.pretoken_set_with_freq: list[tuple[Tokens, int]] = []
        # self.byte_pairs: dict[tuple[bytes, bytes], int] = {}

    def get_pretokenized_corpus(self) -> list[Tokens]:
        #each pre token is a tuple of bytes
        pre_tokenized: list[Tokens] = []
        for x in self.input.split():
            # pretoken = tuple([char.encode("utf-8") for char in x])
            pretoken = tuple(byte.to_bytes() for byte in x.encode("utf-8"))
            pre_tokenized.append(pretoken)
        return pre_tokenized

    def generate_pretoken_set(self, pre_tokenized_corpus: list[Tokens]) -> list[tuple[Tokens, int]]:
        pretoken_set_with_freq: list[tuple[Tokens, int]] = []
        counter: dict[Tokens, int] = {}
        for pretoken in pre_tokenized_corpus:
            counter[pretoken] = counter.get(pretoken, 0) + 1
        for k, v in counter.items():
            pretoken_set_with_freq.append((k,v))
        return pretoken_set_with_freq
    
    def generate_byte_pairs(self, pretoken_set_with_freq: list[tuple[Tokens, int]]) -> dict[BytePair, int]:
        byte_pairs: dict[BytePair, int] = {}
        for item in pretoken_set_with_freq:
            pretoken = item[0]
            for i in range(len(pretoken) - 1):
                pair = (pretoken[i], pretoken[i+1])
                occurences = item[1]
                byte_pairs[pair] = byte_pairs.get(pair, 0) + occurences
        return byte_pairs

    def merge(self, byte_pairs: dict[BytePair, int], freq_map: list[tuple[Tokens, int]], pair: tuple[bytes, bytes]):
        pair_as_bytes = pair[0] + pair[1]
        self.extended_vocab.append(pair_as_bytes)
        for i in range(len(freq_map)):
            j = 0
            while j < len(freq_map[i][0])-1:
                pretoken, _ = freq_map[i]
                if (pretoken[j], pretoken[j+1]) == pair:
                    freq = freq_map[i][1]
                    freq_map[i] = ((pretoken[:j] + (pair_as_bytes,) + pretoken[j+2:]), freq)
                    if j > 0:
                        byte_pairs[(pretoken[j-1], pretoken[j])] -= freq
                        byte_pairs[(pretoken[j-1], pair_as_bytes)] = byte_pairs.get((pretoken[j-1], pair_as_bytes), 0) + freq
                    if j+2 < len(pretoken):
                        byte_pairs[(pretoken[j+1], pretoken[j+2])] -= freq
                        byte_pairs[(pair_as_bytes, pretoken[j+2])] = byte_pairs.get((pair_as_bytes, pretoken[j+2]), 0) + freq
                j += 1
        del byte_pairs[pair]

    def train(self):
        pre_tokenized_corpus = self.get_pretokenized_corpus()
        pretoken_set = self.generate_pretoken_set(pre_tokenized_corpus)
        byte_pairs = self.generate_byte_pairs(pretoken_set)
        arg_max = max(((freq, pair) for pair, freq in byte_pairs.items()), default=None)
        for _ in range(self.max_vocab_size):
            if not arg_max:
                print("no arg max!")
                break
            if arg_max[0] == 0:
                break
            self.merge(byte_pairs, pretoken_set, arg_max[1])
            arg_max = max(((freq, pair) for pair, freq in byte_pairs.items()), default=None)
        self.dump("final", pretoken_set, byte_pairs)

        # print("")
        # for pair in self.byte_pairs:
        #     print(bytes(pair).decode("utf-8"), ":", self.byte_pairs[pair])

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
        print(f"vocab: {' | '.join(show(token) for token in self.extended_vocab)}")
    # def merge(self):



content = Path("input.txt").read_text(encoding="utf-8")
tokenizer = Tokenizer(content, 5)
tokenizer.train()
# string = "hello操"
# print([x for x in string.encode()])
# print([x.encode() for x in string])
# print("hello操".encode())
