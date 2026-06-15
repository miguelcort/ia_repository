"""
Lección: 30-bpe-tokenizer-from-scratch
Fase: 19
Capstone de ingeniería AI: 30 Bpe Tokenizer From Scratch.
"""
from __future__ import annotations
import sys

def bpe_train(corpus, vocab_size):
    """Train BPE: counts -> merge -> vocab."""
    import collections
    vocab = {tuple(word) + ("</w>",): count
            for word, count in corpus.items()}
    merges = []
    while len(vocab) < vocab_size:
        pairs = collections.Counter()
        for word, count in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += count
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_pair(best, vocab)
    return vocab, merges



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
