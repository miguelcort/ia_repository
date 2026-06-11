"""
Lección: 19-tokenizacion-de-subpalabras
Fase: 05
Prerrequisitos: 18-nlp-multilingue
"""
from __future__ import annotations
import sys
import re
from collections import Counter, defaultdict


def tokenizar_palabras(corpus):
    """Tokenizacion por palabras: split por espacios + puntuacion."""
    return re.findall(r"\b\w+\b|[^\w\s]", corpus.lower())


def bpe_train(corpus, num_merges=10, vocabulario_inicial=None):
    """Entrena BPE: iterativamente merge del par mas frecuente.
    Devuelve reglas de merge y vocabulario."""
    # Inicializar tokens con chars + </w>
    palabras = corpus.lower().split()
    vocab = Counter(palabras)
    splits = {p: list(p) + ["</w>"] for p in vocab}
    if vocabulario_inicial is not None:
        alphabet = vocabulario_inicial
    else:
        alphabet = set()
        for p in vocab:
            for c in p:
                alphabet.add(c)
        alphabet = sorted(alphabet)
    vocab_set = set(alphabet)
    merges = []
    for _ in range(num_merges):
        # Contar pares
        pares = defaultdict(int)
        for palabra, split in splits.items():
            for i in range(len(split) - 1):
                pares[(split[i], split[i + 1])] += vocab[palabra]
        if not pares:
            break
        mejor = max(pares, key=pares.get)
        # Merge
        new_splits = {}
        for palabra, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i + 1]) == mejor:
                    new_split.append("".join(mejor))
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[palabra] = new_split
        splits = new_splits
        vocab_set.add("".join(mejor))
        merges.append(mejor)
    return merges, vocab_set


def bpe_apply(texto, merges):
    """Aplica las reglas de merge a un texto."""
    palabras = texto.lower().split()
    tokens = []
    for p in palabras:
        split = list(p) + ["</w>"]
        for a, b in merges:
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(a + b)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            split = new_split
        tokens.extend(split)
    return tokens


def wordpiece_tokenizar(texto, vocab, max_chars=100):
    """WordPiece greedy: longest prefix match.
    Unknown -> [UNK]."""
    palabras = texto.lower().split()
    tokens = []
    for palabra in palabras:
        start = 0
        sub_tokens = []
        while start < len(palabra):
            end = len(palabra)
            cur = None
            while end > start:
                substr = palabra[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in vocab:
                    cur = substr
                    break
                end -= 1
            if cur is None:
                sub_tokens.append("[UNK]")
                break
            sub_tokens.append(cur)
            start = end
        tokens.extend(sub_tokens)
    return tokens


def main() -> int:
    corpus = "el gato come pescado el perro come carne el gato come"
    merges, vocab = bpe_train(corpus, num_merges=10)
    print(f"Merges: {merges}")
    print(f"Vocab size: {len(vocab)}")
    texto = "el gato come"
    tokens = bpe_apply(texto, merges)
    print(f"BPE tokens: {tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())