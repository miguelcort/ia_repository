"""
Lección: 02-bolsa-de-palabras-y-tfidf
Fase: 05
Prerrequisitos: 01-procesamiento-de-texto
"""
from __future__ import annotations
import sys
import re
import math
import numpy as np
from collections import Counter


def tokenizar(texto, patron=r"\b\w+\b"):
    """Tokeniza usando regex."""
    return re.findall(patron, texto.lower())


def construir_vocabulario(docs, min_freq=1):
    """Construye vocabulario: palabra -> indice. Solo palabras con freq >= min_freq."""
    counter = Counter()
    for doc in docs:
        counter.update(set(tokenizar(doc)))
    palabras = [w for w, c in counter.items() if c >= min_freq]
    palabras.sort()
    return {w: i for i, w in enumerate(palabras)}


def bag_of_words(doc, vocab):
    """Vector BoW: cuenta ocurrencias de cada palabra del vocab."""
    tokens = tokenizar(doc)
    vector = np.zeros(len(vocab), dtype=np.int32)
    for t in tokens:
        if t in vocab:
            vector[vocab[t]] += 1
    return vector


def tf_idf(docs, vocab):
    """Calcula matriz TF-IDF: cada fila es un documento, cada columna una palabra.
    TF(t, d) = count(t, d) / |d|.
    IDF(t) = log(N / (1 + df(t))).
    TF-IDF = TF * IDF.
    """
    N = len(docs)
    # Term frequency por documento
    tf_matrix = np.zeros((N, len(vocab)))
    for i, doc in enumerate(docs):
        tokens = tokenizar(doc)
        if len(tokens) == 0:
            continue
        for t in tokens:
            if t in vocab:
                tf_matrix[i, vocab[t]] += 1
        tf_matrix[i] /= len(tokens)
    # Document frequency
    df = np.zeros(len(vocab))
    for doc in docs:
        for w in set(tokenizar(doc)):
            if w in vocab:
                df[vocab[w]] += 1
    idf = np.log(N / (1.0 + df)) + 1.0  # smoothed
    return tf_matrix * idf


def cosine_similarity(a, b):
    """Similitud coseno entre dos vectores."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main() -> int:
    docs = [
        "el gato negro salta sobre el perro",
        "el perro ladra al gato",
        "los pajaros vuelan alto",
    ]
    vocab = construir_vocabulario(docs)
    print(f"Vocabulario: {vocab}")
    bows = np.array([bag_of_words(d, vocab) for d in docs])
    print(f"BoW shape: {bows.shape}")
    print(f"BoW[0]: {bows[0]}")
    tfidf = tf_idf(docs, vocab)
    print(f"TF-IDF shape: {tfidf.shape}")
    # Similitud doc 0 vs 1 (ambos tienen gato/perro)
    sim = cosine_similarity(tfidf[0], tfidf[1])
    print(f"Similitud doc 0-1: {sim:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())