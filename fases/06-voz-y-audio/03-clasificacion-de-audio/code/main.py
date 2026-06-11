"""
Lección: 03-clasificacion-de-audio
Fase: 06
Prerrequisitos: 02-espectrogramas-y-caracteristicas-mel
"""
from __future__ import annotations
import sys
import numpy as np


def mock_classifier(features, pesos, bias):
    """Mock: clasificador lineal para features de audio.
    features: (N, D). pesos: (D, n_clases). bias: (n_clases,)."""
    return features @ pesos + bias


def softmax(z):
    z_est = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z_est)
    return exp / exp.sum(axis=-1, keepdims=True)


def label_clase(logits, etiquetas):
    """Devuelve la clase con mayor probabilidad."""
    probs = softmax(logits)
    idx = int(probs.argmax())
    return etiquetas[idx], float(probs[idx])


def accuracy(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def main() -> int:
    rng = np.random.default_rng(0)
    N, D, n_clases = 10, 80, 5  # 80 features (e.g. mel bins), 5 clases
    features = rng.normal(size=(N, D))
    pesos = rng.normal(scale=0.1, size=(D, n_clases))
    bias = np.zeros(n_clases)
    logits = mock_classifier(features, pesos, bias)
    etiquetas = ["musica", "voz", "perro", "alarma", "silencio"]
    y_pred = [label_clase(logits[i], etiquetas)[0] for i in range(N)]
    y_true = rng.choice(etiquetas, size=N)
    acc = accuracy(y_true, y_pred)
    print(f"Accuracy: {acc:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())