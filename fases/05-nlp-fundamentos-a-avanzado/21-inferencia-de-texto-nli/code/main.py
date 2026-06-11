"""
Lección: 21-inferencia-de-texto-nli
Fase: 05
Prerrequisitos: 20-salidas-estructuradas-y-decoding-constrenido
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(z):
    z_est = z - z.max()
    exp = np.exp(z_est)
    return exp / exp.sum()


def nli_3way_mock(premisa, hipotesis):
    """Mock: clasifica la relacion entre premisa e hipotesis.
    0: entailment, 1: neutral, 2: contradiction.
    """
    p_words = set(premisa.lower().split())
    h_words = set(hipotesis.lower().split())
    if h_words.issubset(p_words):
        return 0  # entailment
    if p_words & h_words:
        return 1  # neutral
    return 2  # contradiction


def contradiction_loss(y_true, logits):
    """Cross-entropy para NLI: y_true es one-hot (3,), logits (3,)."""
    probs = softmax(logits)
    eps = 1e-9
    return float(-np.sum(y_true * np.log(probs + eps)))


def accuracy_nli(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def main() -> int:
    # Ejemplo
    premisa = "el gato come pescado"
    hipotesis = "el gato come"
    rel = nli_3way_mock(premisa, hipotesis)
    print(f"Premisa: {premisa}")
    print(f"Hipotesis: {hipotesis}")
    print(f"Relacion: {rel} (0=entailment, 1=neutral, 2=contradiction)")
    return 0


if __name__ == "__main__":
    sys.exit(main())