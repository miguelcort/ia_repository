"""
Lección: 11-traduccion-automatica
Fase: 05
Prerrequisitos: 10-mecanismo-de-atencion
"""
from __future__ import annotations
import sys
import numpy as np


def bleu_score(reference, candidate, max_n=4):
    """BLEU: precision multiplicativa de n-gramas hasta max_n,
    con brevity penalty.
    """
    from collections import Counter
    def n_grams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    # Brevity penalty
    if len(candidate) == 0:
        return 0.0
    bp = min(1.0, np.exp(1 - len(reference) / len(candidate)))

    log_bleu = 0.0
    n_usados = 0
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(n_grams(reference, n))
        cand_ngrams = Counter(n_grams(candidate, n))
        total_cand = sum(cand_ngrams.values())
        if total_cand == 0:
            # No hay n-gramas de este tamano, ignorar
            continue
        clipped = {k: min(c, ref_ngrams[k]) for k, c in cand_ngrams.items()}
        total_clipped = sum(clipped.values())
        precision = total_clipped / total_cand
        log_bleu += np.log(precision + 1e-9)
        n_usados += 1
    if n_usados == 0:
        return 0.0
    log_bleu /= n_usados
    return float(bp * np.exp(log_bleu))


def tokenizar(texto):
    return texto.lower().split()


def main() -> int:
    ref = tokenizar("el gato come pescado")
    cand = tokenizar("el gato come pescado")
    print(f"BLEU perfecto: {bleu_score(ref, cand):.3f}")
    cand2 = tokenizar("el gato come")
    print(f"BLEU parcial: {bleu_score(ref, cand2):.3f}")
    cand3 = tokenizar("el perro ladra fuerte")
    print(f"BLEU sin match: {bleu_score(ref, cand3):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())