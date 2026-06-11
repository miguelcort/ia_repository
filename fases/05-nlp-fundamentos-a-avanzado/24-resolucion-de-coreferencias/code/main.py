"""
Lección: 24-resolucion-de-coreferencias
Fase: 05
Prerrequisitos: 23-estrategias-de-chunking-y-rag
"""
from __future__ import annotations
import sys
import numpy as np


def resolver_coreferencias_mock(oracion, mencions):
    """Mock: resuelve que mencion se refiere a que entidad.
    mencions: lista de strings. Devuelve dict {mencion: entidad}.
    Estrategia simple: misma entidad si comparten nombre/sinonimo.
    """
    entidades = {}
    # Mock: cada mencion unica -> una entidad
    for i, m in enumerate(mencions):
        entidades[i] = {"mencion": m, "entidad_id": f"ent_{i}"}
    return entidades


def distancia_entre_menciones(m1, m2):
    """Mock: distancia basada en numero de tokens entre dos menciones."""
    return abs(m1 - m2)  # en tokens


def pairwise_score(m1, m2, embedding):
    """Score de coreferencia: misma entidad si similitud embedding alta.
    Mock: cos similarity."""
    if np.linalg.norm(embedding[m1]) == 0 or np.linalg.norm(embedding[m2]) == 0:
        return 0.0
    return float(embedding[m1] @ embedding[m2] / (np.linalg.norm(embedding[m1]) * np.linalg.norm(embedding[m2])))


def mention_pair_score(m1_text, m2_text, emb):
    """Score entre dos menciones basado en embeddings."""
    return float(emb[m1_text] @ emb[m2_text] / (np.linalg.norm(emb[m1_text]) * np.linalg.norm(emb[m2_text])) + 1e-9)


def main() -> int:
    mencions = ["Maria", "ella", "la nina", "Pedro", "el"]
    entidades = resolver_coreferencias_mock("Maria trabajo con Pedro y ella estudio", mencions)
    print(f"Entidades: {entidades}")
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(5, 16))
    s = mention_pair_score(0, 1, emb)
    print(f"Similitud 0-1: {s:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())