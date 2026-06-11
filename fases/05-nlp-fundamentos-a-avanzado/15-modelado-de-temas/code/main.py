"""
Lección: 15-modelado-de-temas
Fase: 05
Prerrequisitos: 14-recuperacion-de-informacion
"""
from __future__ import annotations
import sys
import numpy as np


def tokenizar(texto):
    import re
    return re.findall(r"\b\w+\b", texto.lower())


def lda_gibbs_step(docs, k_topics, n_iter=20, semilla=0):
    """LDA simplificado con collapsed Gibbs sampling.
    Asume: cada doc tiene asignado un topic por token.
    Devuelve distribuciones de topics por doc y word por topic.
    """
    rng = np.random.default_rng(semilla)
    # Tokenize
    docs_tokens = [tokenizar(d) for d in docs]
    # Asignar topics iniciales
    z = [[rng.integers(0, k_topics) for _ in tokens] for tokens in docs_tokens]
    # Conteos
    V = len(set(t for tokens in docs_tokens for t in tokens))
    n_dk = np.zeros((len(docs), k_topics))  # doc-topic
    n_kw = np.zeros((k_topics, V))  # topic-word
    n_k = np.zeros(k_topics)
    vocab = sorted(set(t for tokens in docs_tokens for t in tokens))
    word2id = {w: i for i, w in enumerate(vocab)}
    # Init counts
    for d, tokens in enumerate(docs_tokens):
        for w_idx, t in enumerate(tokens):
            wid = word2id[t]
            k = z[d][w_idx]
            n_dk[d, k] += 1
            n_kw[k, wid] += 1
            n_k[k] += 1
    alpha = 0.1
    beta = 0.1
    for it in range(n_iter):
        for d, tokens in enumerate(docs_tokens):
            for w_idx, t in enumerate(tokens):
                wid = word2id[t]
                k_old = z[d][w_idx]
                # Quitar
                n_dk[d, k_old] -= 1
                n_kw[k_old, wid] -= 1
                n_k[k_old] -= 1
                # Sample nuevo k
                probs = (n_dk[d] + alpha) * (n_kw[:, wid] + beta) / (n_k + V * beta)
                probs /= probs.sum()
                k_new = int(rng.choice(k_topics, p=probs))
                z[d][w_idx] = k_new
                n_dk[d, k_new] += 1
                n_kw[k_new, wid] += 1
                n_k[k_new] += 1
    # Normalizar
    theta = n_dk / n_dk.sum(axis=1, keepdims=True)  # doc-topic
    phi = n_kw / n_kw.sum(axis=1, keepdims=True)  # topic-word
    return theta, phi, vocab


def top_palabras_por_topic(phi, vocab, top_n=3):
    """Para cada topic, devuelve las top_n palabras mas probables."""
    result = {}
    for k in range(phi.shape[0]):
        idx = np.argsort(phi[k])[::-1][:top_n]
        result[k] = [vocab[i] for i in idx]
    return result


def main() -> int:
    docs = [
        "el gato come pescado",
        "el perro come carne",
        "el gato come carne",
        "el gato come",
    ]
    theta, phi, vocab = lda_gibbs_step(docs, k_topics=2, n_iter=50, semilla=42)
    print(f"Doc-topic shape: {theta.shape}")
    print(f"Topic-word shape: {phi.shape}")
    top = top_palabras_por_topic(phi, vocab, top_n=3)
    for k, palabras in top.items():
        print(f"Topic {k}: {palabras}")
    return 0


if __name__ == "__main__":
    sys.exit(main())