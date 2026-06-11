"""
Lección: 14-recuperacion-de-informacion
Fase: 05
Prerrequisitos: 13-preguntas-y-respuestas
"""
from __future__ import annotations
import sys
import math
import re
import numpy as np
from collections import Counter


def tokenizar(texto):
    return re.findall(r"\b\w+\b", texto.lower())


def construir_indice_invertido(docs):
    """Indice invertido: termino -> {doc_id: [posiciones]}."""
    indice = {}
    for doc_id, doc in enumerate(docs):
        for pos, t in enumerate(tokenizar(doc)):
            if t not in indice:
                indice[t] = {}
            if doc_id not in indice[t]:
                indice[t][doc_id] = []
            indice[t][doc_id].append(pos)
    return indice


def bm25(query, docs, indice, k1=1.5, b=0.75):
    """BM25 score. Clasico en IR.
    BM25(d, q) = sum IDF(t) * (f(t, d) * (k1 + 1)) / (f(t, d) + k1 * (1 - b + b * |d| / avgdl))
    """
    N = len(docs)
    doc_lens = [len(tokenizar(d)) for d in docs]
    avgdl = sum(doc_lens) / N if N > 0 else 1.0
    q_tokens = tokenizar(query)
    scores = np.zeros(N)
    for t in q_tokens:
        if t not in indice:
            continue
        # df
        df = len(indice[t])
        # IDF (Robertson)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        for doc_id, posiciones in indice[t].items():
            f = len(posiciones)
            denom = f + k1 * (1 - b + b * doc_lens[doc_id] / avgdl)
            scores[doc_id] += idf * f * (k1 + 1) / denom
    return scores


def precision_at_k(relevantes, retrieved, k):
    retrieved_set = set(int(x) for x in (retrieved[:k] if hasattr(retrieved[:k], 'tolist') else retrieved[:k]))
    relevantes_set = set(int(x) for x in relevantes)
    if not retrieved_set:
        return 0.0
    hits = sum(1 for r in retrieved_set if r in relevantes_set)
    return hits / len(retrieved_set)


def main() -> int:
    docs = [
        "el gato come pescado en la cocina",
        "el perro ladra fuerte en el jardin",
        "los pajaros vuelan alto en el cielo",
    ]
    indice = construir_indice_invertido(docs)
    print(f"Terminos en indice: {len(indice)}")
    scores = bm25("gato cocina", docs, indice)
    ranking = np.argsort(scores)[::-1]
    print(f"Scores: {scores.round(3)}")
    print(f"Ranking: {ranking}")
    p_at_1 = precision_at_k([0], ranking, 1)
    print(f"P@1 (doc 0 relevante): {p_at_1:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())