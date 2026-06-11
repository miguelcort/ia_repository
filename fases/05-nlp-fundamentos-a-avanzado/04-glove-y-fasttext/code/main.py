"""
Lección: 04-glove-y-fasttext
Fase: 05
Prerrequisitos: 03-embeddings-de-palabras-word2vec
"""
from __future__ import annotations
import sys
import numpy as np


def coocurrencia(corpus, vocab, ventana=2):
    """Cuenta co-ocurrencias de palabras en una ventana."""
    V = len(vocab)
    M = np.zeros((V, V), dtype=np.float32)
    for oracion in corpus:
        tokens = [vocab[t] for t in oracion if t in vocab]
        for i, ti in enumerate(tokens):
            for j in range(max(0, i - ventana), min(len(tokens), i + ventana + 1)):
                if i != j:
                    M[ti, tokens[j]] += 1
    return M


def glove_loss(M, W, b_i, b_j, vocab_size, embedding_dim, x_max=100, alpha=0.75):
    """Loss GloVe simplificado. M es la matriz de co-ocurrencia.
    Loss = sum f(M_ij) * (W_i . W_j + b_i + b_j - log(M_ij))^2.
    f(x) = (x/x_max)^alpha si x < x_max, sino 1.
    """
    total = 0.0
    for i in range(vocab_size):
        for j in range(vocab_size):
            x = M[i, j]
            if x == 0:
                continue
            f = (x / x_max) ** alpha if x < x_max else 1.0
            pred = W[i] @ W[j] + b_i[i] + b_j[j]
            log_x = np.log(x)
            total += f * (pred - log_x) ** 2
    return float(total)


def fasttext_subword(word, n_min=3, n_max=6):
    """Genera subword n-grams + la palabra completa.
    'apple' con n=3 -> ['<ap', 'app', 'ppl', 'ple', 'le>', 'apple'].
    """
    if not word:
        return []
    padded = "<" + word + ">"
    subs = [padded]  # la palabra completa
    for n in range(n_min, min(n_max + 1, len(padded))):
        for i in range(len(padded) - n + 1):
            subs.append(padded[i:i + n])
    return subs


def embedding_desde_subwords(subwords, vocab_subword, dim=100, semilla=0):
    """Embedding fastText: promedio de los embeddings de los subwords."""
    rng = np.random.default_rng(semilla)
    W = rng.normal(0, 0.1, size=(len(vocab_subword), dim))
    embeddings = []
    for w in subwords:
        if w in vocab_subword:
            embeddings.append(W[vocab_subword[w]])
    if not embeddings:
        return np.zeros(dim)
    return np.mean(embeddings, axis=0)


def main() -> int:
    corpus = [["el", "gato", "come", "pescado"],
              ["el", "perro", "ladra", "al", "gato"]]
    vocab = {"el": 0, "gato": 1, "come": 2, "pescado": 3, "perro": 4, "ladra": 5, "al": 6}
    M = coocurrencia(corpus, vocab, ventana=2)
    print(f"Co-ocurrencia shape: {M.shape}, total: {M.sum()}")
    # FastText
    subwords = fasttext_subword("apple", n_min=3, n_max=5)
    print(f"Subwords de 'apple': {subwords}")
    return 0


if __name__ == "__main__":
    sys.exit(main())