"""
Lección: 16-generacion-de-texto-pre-transformer
Fase: 05
Prerrequisitos: 15-modelado-de-temas
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(z):
    z_est = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z_est)
    return exp / exp.sum(axis=-1, keepdims=True)


def greedy_decode(logits):
    """Greedy: token con max prob en cada paso."""
    return int(np.argmax(logits))


def temperature_scale(logits, T=1.0):
    """Escala logits por temperatura T. T<1: mas sharp. T>1: mas diverse."""
    return logits / T


def top_k_filter(logits, k):
    """Mask logits fuera del top-k. Logits -> -inf para los demas."""
    threshold = np.sort(logits)[-k]
    return np.where(logits >= threshold, logits, -1e9)


def nucleus_filter(logits, p=0.9):
    """Top-p (nucleus) sampling: conservar tokens hasta cumulativa p."""
    probs = softmax(logits)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumul = np.cumsum(sorted_probs)
    # Encuentra el primer indice donde cumul > p
    cutoff = np.searchsorted(cumul, p)
    if cutoff == 0:
        cutoff = 1
    keep_idx = sorted_idx[:cutoff + 1]
    new_probs = np.zeros_like(probs)
    new_probs[keep_idx] = probs[keep_idx]
    new_probs /= new_probs.sum()
    return new_probs


def beam_search(logits_seq, beam_width=3):
    """Mock de beam search: mantener top-k secuencias parciales.
    logits_seq: (T, V). Devuelve la mejor secuencia de indices."""
    T, V = logits_seq.shape
    # Cada beam es una tupla (log_prob, [indices])
    beams = [(0.0, [])]
    for t in range(T):
        nuevos = []
        for log_prob, seq in beams:
            for v in range(V):
                new_seq = seq + [v]
                new_log_prob = log_prob + np.log(softmax(logits_seq[t])[v] + 1e-9)
                nuevos.append((new_log_prob, new_seq))
        # Top-k por log_prob
        nuevos.sort(key=lambda x: -x[0])
        beams = nuevos[:beam_width]
    # Devolver la mejor
    return beams[0][1]


def main() -> int:
    logits = np.array([2.0, 1.0, 0.5, 0.1])
    print(f"Greedy: {greedy_decode(logits)}")
    print(f"Temperature 0.5: {softmax(temperature_scale(logits, 0.5))}")
    print(f"Top-k=2 filter: {top_k_filter(logits, 2)}")
    print(f"Nucleus p=0.9: {nucleus_filter(logits, 0.9).round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())