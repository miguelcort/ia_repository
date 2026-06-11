"""
Lección: 18-multi-token-prediction
Fase: 10
Multi-Token Prediction (MTP, Meta 2024): predecir multiples tokens futuros.
Training signal +4x, 2-3x inference speedup con spec decoding.
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_loss(logits, targets):
    """logits: (n, vocab). targets: (n,)."""
    p = softmax(logits, axis=-1)
    p = np.clip(p, 1e-9, 1.0)
    return float(-np.log(p[np.arange(len(targets)), targets]).mean())


def mtp_loss(main_logits, aux_logits, targets, aux_targets, aux_weight=0.3):
    """Multi-token prediction loss.
    L = CE(main) + aux_weight * CE(aux)
    """
    main_loss = cross_entropy_loss(main_logits, targets)
    aux_loss = cross_entropy_loss(aux_logits, aux_targets)
    return main_loss + aux_weight * aux_loss


def mtp_training_signal_gain(k_aux=4):
    """Training signal: cada token provee info para k+1 predicciones.
    Densidad de signal: 1+k / 1 = 1+k = 5x con k=4.
    """
    return 1 + k_aux


def mtp_inference_speedup(aux_acceptance_rate=0.7, k_aux=4):
    """MTP + speculative decoding: aux predictions como drafts.
    Speedup = (k + 1) / (k * cost_draft + 1) con acceptance r.
    """
    expected_accepted = sum(aux_acceptance_rate ** (i + 1) for i in range(k_aux))
    cost = 0.1  # aux heads costo bajo
    return (expected_accepted + 1) / (k_aux * cost + 1)


def mtp_components():
    """Componentes de MTP."""
    return {
        "Main head": "Predice t+1 (next token)",
        "Aux heads": "k heads predicen t+2, t+3, t+4, t+5",
        "Shared trunk": "Transformer compartido",
        "Loss": "Suma ponderada de k+1 losses",
        "Training": "+training signal density, +sample efficiency",
        "Inference": "Aux heads como drafts para spec decode",
    }


def main() -> int:
    # MTP gain
    g = mtp_training_signal_gain(k_aux=4)
    print(f"MTP training signal gain (k=4): {g}x")
    # MTP speedup
    sp = mtp_inference_speedup(aux_acceptance_rate=0.7, k_aux=4)
    print(f"MTP inference speedup: {sp:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())