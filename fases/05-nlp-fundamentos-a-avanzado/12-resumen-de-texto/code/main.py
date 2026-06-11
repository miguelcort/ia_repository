"""
Lección: 12-resumen-de-texto
Fase: 05
Prerrequisitos: 11-traduccion-automatica
"""
from __future__ import annotations
import sys
import numpy as np


def tokenizar(texto):
    return texto.lower().split()


def rouge_n(reference, candidate, n=1):
    """ROUGE-N: recall de n-gramas.
    ROUGE = sum(min(count(cand_ngram), count(ref_ngram))) / sum(count(ref_ngram)).
    """
    from collections import Counter
    def n_grams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    ref_ngrams = Counter(n_grams(reference, n))
    cand_ngrams = Counter(n_grams(candidate, n))
    if not ref_ngrams:
        return 0.0
    clipped = {k: min(c, ref_ngrams[k]) for k, c in cand_ngrams.items()}
    overlap = sum(clipped.values())
    return overlap / sum(ref_ngrams.values())


def rouge_l(reference, candidate):
    """ROUGE-L: Longest Common Subsequence. Recall = LCS/ref, Precision = LCS/cand."""
    m, n = len(reference), len(candidate)
    if m == 0 or n == 0:
        return 0.0
    # LCS dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i - 1] == candidate[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    recall = lcs / m
    precision = lcs / n
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    return f1


def extractive_lead_n(texto, n_oraciones=2):
    """Baseline extractivo: primeras n oraciones."""
    import re
    ors = re.split(r'(?<=[.!?])\s+', texto)
    return " ".join(ors[:n_oraciones])


def main() -> int:
    ref = tokenizar("el gato negro come pescado en la cocina")
    cand = tokenizar("el gato come pescado")
    print(f"ROUGE-1: {rouge_n(ref, cand, 1):.3f}")
    print(f"ROUGE-2: {rouge_n(ref, cand, 2):.3f}")
    print(f"ROUGE-L: {rouge_l(ref, cand):.3f}")
    texto = "El gato come. El perro ladra. Los pajaros vuelan. El sol brilla."
    print(f"Lead-2: {extractive_lead_n(texto, 2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())