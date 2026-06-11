"""
Lección: 18-nlp-multilingue
Fase: 05
Prerrequisitos: 17-chatbots-de-reglas-a-neuronal
"""
from __future__ import annotations
import sys
import numpy as np


def langdetect_mock(texto):
    """Mock: deteccion de idioma por caracteres unicos."""
    texto_lower = texto.lower()
    scores = {
        "es": 0.0, "en": 0.0, "fr": 0.0, "de": 0.0, "pt": 0.0,
    }
    # Caracteristicas tipicas
    for c in "áéíóúñ":
        if c in texto_lower:
            scores["es"] += 1
    for c in "àâçèêëîïôùû":
        if c in texto_lower:
            scores["fr"] += 1
    for c in "äöüß":
        if c in texto_lower:
            scores["de"] += 1
    for c in "ãõáéíóú":
        if c in texto_lower:
            scores["pt"] += 1
    for c in "abcdefghijklmnopqrstuvwxyz":
        if c in texto_lower:
            scores["en"] += 0.1
    mejor = max(scores, key=scores.get)
    return mejor, scores[mejor]


def align_palabras(src_tokens, tgt_tokens):
    """Alineacion de palabras basica: longest common subsequence.
    Devuelve pares (src_idx, tgt_idx)."""
    m, n = len(src_tokens), len(tgt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_tokens[i - 1] == tgt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # Backtrack
    pares = []
    i, j = m, n
    while i > 0 and j > 0:
        if src_tokens[i - 1] == tgt_tokens[j - 1]:
            pares.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    pares.reverse()
    return pares


def main() -> int:
    es = "Hola, como estas?"
    en = "Hello, how are you?"
    print(f"es: {langdetect_mock(es)}")
    print(f"en: {langdetect_mock(en)}")
    # Alignment
    src = ["el", "gato", "come", "pescado"]
    tgt = ["the", "cat", "eats", "fish"]
    pares = align_palabras(src, tgt)
    print(f"Alineacion: {pares}")
    return 0


if __name__ == "__main__":
    sys.exit(main())