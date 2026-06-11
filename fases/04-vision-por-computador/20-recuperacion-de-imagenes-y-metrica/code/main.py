"""
Lección: 20-recuperacion-de-imagenes-y-metrica
Fase: 04
Prerrequisitos: 18-clip-vocabulario-abierto
"""
from __future__ import annotations
import sys
import numpy as np


def encoder_mock(img, dim=128, semilla=0):
    """Mock: convierte imagen en embedding."""
    H, W = img.shape[:2]
    rng = np.random.default_rng(semilla + H * W)
    return rng.normal(0, 1, size=(dim,))


def l2_normalize(x):
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n


def construir_indice(embeddings):
    """Indexa embeddings normalizados para busqueda eficiente."""
    norm = np.array([l2_normalize(e) for e in embeddings])
    return norm


def buscar_por_similitud(query, indice, top_k=5):
    """Top-k mas similares a query."""
    q = l2_normalize(query)
    sims = indice @ q
    idx = np.argsort(sims)[::-1][:top_k]
    return idx, sims[idx]


def _to_set(seq, k=None):
    """Convierte secuencia (list o array) a set, opcionalmente truncada a k."""
    if k is not None:
        seq = seq[:k]
    if hasattr(seq, 'tolist'):
        seq = seq.tolist()
    return set(int(x) for x in seq)


def recall_at_k(relevantes, retrieved, k):
    """Recall@k: proporcion de relevantes que aparecen en top-k retrieved."""
    if not relevantes:
        return 0.0
    retrieved_set = _to_set(retrieved, k)
    relevantes_set = _to_set(relevantes)
    hits = sum(1 for r in relevantes_set if r in retrieved_set)
    return hits / len(relevantes_set)


def precision_at_k(relevantes, retrieved, k):
    """Precision@k: proporcion de top-k que son relevantes."""
    retrieved_set = _to_set(retrieved, k)
    relevantes_set = _to_set(relevantes)
    if not retrieved_set:
        return 0.0
    hits = sum(1 for r in retrieved_set if r in relevantes_set)
    return hits / len(retrieved_set)


def mean_average_precision(relevantes_por_query, rankings):
    """MAP: promedio del average precision por query."""
    aps = []
    for rels, ranking in zip(relevantes_por_query, rankings):
        if not rels:
            continue
        hits = 0
        precision_sum = 0.0
        for i, item in enumerate(ranking):
            if item in rels:
                hits += 1
                precision_sum += hits / (i + 1)
        ap = precision_sum / len(rels) if rels else 0.0
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def main() -> int:
    # Simula 100 imagenes indexadas
    np.random.default_rng(0)
    embeddings = [encoder_mock(np.random.default_rng(i).normal(size=(10, 10, 3)), dim=128, semilla=i) for i in range(100)]
    indice = construir_indice(embeddings)
    print(f"Indice: {indice.shape}")
    query = encoder_mock(np.random.default_rng(99).normal(size=(10, 10, 3)), dim=128, semilla=99)
    top5, sims = buscar_por_similitud(query, indice, top_k=5)
    print(f"Top-5 indices: {top5}, sims: {sims.round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())