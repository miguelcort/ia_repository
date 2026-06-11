"""
Lección: 02-construyendo-un-tokenizer
Fase: 10
Construir un BPE tokenizer desde cero: pre-tokenize, learn merges, encode, decode.
"""
from __future__ import annotations
import sys
from collections import Counter, defaultdict
import re


PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|\s+\S|\s+|[^\s]+""")


def pre_tokenize(text):
    """GPT-2 style pre-tokenization."""
    return [m.group(0) for m in PAT.finditer(text)]


def get_pairs(word):
    """Get all adjacent pairs in a word (list of symbols)."""
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class BPETokenizer:
    """BPE tokenizer from scratch."""

    def __init__(self):
        self.vocab = {}  # token -> id
        self.merges = []  # list of (a, b) pairs to merge, in order
        self.cache = {}  # token string -> list of token ids

    def train(self, corpus, vocab_size):
        """Train BPE on corpus.
        corpus: list of strings.
        """
        # Step 1: pre-tokenize and count
        word_freqs = Counter()
        for text in corpus:
            pre_tokens = pre_tokenize(text)
            for token in pre_tokens:
                word_freqs[token] += 1
        # Step 2: init vocab with chars
        chars = set()
        for word in word_freqs:
            for ch in word:
                chars.add(ch)
        self.vocab = {ch: i for i, ch in enumerate(sorted(chars))}
        # Step 3: represent words as list of symbols
        word_syms = {w: list(w) for w in word_freqs}
        # Step 4: merge
        while len(self.vocab) < vocab_size:
            pairs = self._get_pair_freqs(word_syms, word_freqs)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_token = "".join(best)
            if new_token in self.vocab:
                # Already exists, skip
                break
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best)
            # Update words
            new_word_syms = {}
            for w, syms in word_syms.items():
                new_syms = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                        new_syms.append(syms[i] + syms[i + 1])
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                new_word_syms[w] = new_syms
            word_syms = new_word_syms

    def _get_pair_freqs(self, word_syms, word_freqs):
        """Count pair frequencies weighted by word freq."""
        pairs = defaultdict(int)
        for w, syms in word_syms.items():
            for i in range(len(syms) - 1):
                pairs[(syms[i], syms[i + 1])] += word_freqs[w]
        return pairs

    def encode(self, text):
        """Encode text to token ids."""
        if text in self.cache:
            return self.cache[text]
        ids = []
        for token in pre_tokenize(text):
            # Apply merges in order
            syms = list(token)
            for a, b in self.merges:
                new_syms = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == (a, b):
                        new_syms.append(syms[i] + syms[i + 1])
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                syms = new_syms
            for s in syms:
                if s in self.vocab:
                    ids.append(self.vocab[s])
                else:
                    # Fallback: char-level
                    for ch in s:
                        if ch in self.vocab:
                            ids.append(self.vocab[ch])
        self.cache[text] = ids
        return ids

    def decode(self, ids):
        """Decode token ids to text."""
        id_to_token = {i: t for t, i in self.vocab.items()}
        return "".join(id_to_token.get(i, "") for i in ids)


def main() -> int:
    corpus = [
        "Hola mundo, esto es un test.",
        "El gato come pescado.",
        "El perro corre rapido.",
    ]
    tokenizer = BPETokenizer()
    tokenizer.train(corpus, vocab_size=50)
    print(f"Vocab size: {len(tokenizer.vocab)}")
    encoded = tokenizer.encode("El gato come.")
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())