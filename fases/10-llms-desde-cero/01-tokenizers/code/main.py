"""
Lección: 01-tokenizers
Fase: 10
Tokenizers: BPE, WordPiece, SentencePiece, Unigram.
Vocabulary, encoding, decoding, special tokens.
"""
from __future__ import annotations
import sys
import numpy as np


def char_level_tokenize(text):
    """Char-level tokenization."""
    return list(text)


def word_level_tokenize(text):
    """Word-level tokenization simple (whitespace + punctuation)."""
    import re
    return re.findall(r"\w+|[^\w\s]", text)


def bpe_merge_pair(pair, vocab):
    """BPE merge: reemplazar pair mas frecuente por nuevo token.
    Simulado: cuenta occurrences y merge.
    """
    counts = {}
    for token in vocab:
        for i in range(len(token) - 1):
            p = (token[i], token[i + 1])
            counts[p] = counts.get(p, 0) + 1
    if not counts:
        return None
    best = max(counts, key=counts.get)
    return best


def compute_pair_frequencies(vocab):
    """Cuenta frequencies de pairs adyacentes.
    vocab: dict token_str -> count.
    Returns: dict pair -> count.
    """
    pairs = {}
    for token, count in vocab.items():
        chars = token.split()
        for i in range(len(chars) - 1):
            pair = (chars[i], chars[i + 1])
            pairs[pair] = pairs.get(pair, 0) + count
    return pairs


def bpe_train_step(vocab, num_merges=10):
    """BPE training: num_merges iteraciones de find best pair y merge."""
    merges = []
    for _ in range(num_merges):
        pairs = compute_pair_frequencies(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        # Merge
        new_vocab = {}
        for token, count in vocab.items():
            new_token = token.replace(f"{best[0]} {best[1]}", f"{best[0]}{best[1]}")
            new_vocab[new_token] = count
        vocab = new_vocab
        merges.append(best)
    return vocab, merges


def tokenize_with_vocab(text, vocab):
    """Tokenize text usando vocab (subword lookup).
    Mock: si token en vocab, use, sino split en chars.
    """
    tokens = []
    i = 0
    while i < len(text):
        # Greedy longest match
        for j in range(len(text), i, -1):
            sub = text[i:j]
            if sub in vocab:
                tokens.append(sub)
                i = j
                break
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def special_tokens_list():
    """Special tokens comunes."""
    return {
        "BOS": "<s>",  # Beginning of sequence
        "EOS": "</s>",  # End of sequence
        "PAD": "<pad>",  # Padding
        "UNK": "<unk>",  # Unknown
        "MASK": "<mask>",  # Mask (BERT)
        "SEP": "[SEP]",  # Separator (BERT)
        "CLS": "[CLS]",  # Classification (BERT)
        "PAD_BPE": "<|endoftext|>",  # GPT
    }


def main() -> int:
    text = "Hola mundo, esto es un test."
    chars = char_level_tokenize(text)
    words = word_level_tokenize(text)
    print(f"Chars: {len(chars)} tokens")
    print(f"Words: {len(words)} tokens: {words}")
    # BPE toy
    vocab = {"h o l a": 1, "m u n d o": 1, "h o l a m u n d o": 1}
    new_vocab, merges = bpe_train_step(vocab, num_merges=5)
    print(f"BPE merges: {merges}")
    return 0


if __name__ == "__main__":
    sys.exit(main())