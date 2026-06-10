"""
Lección: 04-clasificacion-de-imagenes
Fase: 04
Prerrequisitos: 03-cnns-desde-lenet-hasta-resnet
"""
from __future__ import annotations
import sys
import numpy as np


def softmax(z):
    z_est = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z_est)
    return exp / exp.sum(axis=-1, keepdims=True)


def cross_entropy(logits, y_true):
    """CCE: -sum(y * log(softmax(logits)))."""
    probs = softmax(logits)
    eps = 1e-9
    return float(-np.mean(np.sum(y_true * np.log(probs + eps), axis=-1)))


def top_k_accuracy(logits, y_true, k=5):
    """Top-k accuracy: 1 si la clase verdadera esta en top-k predicciones."""
    probs = softmax(logits)
    top_k = np.argsort(probs, axis=-1)[:, -k:]
    if y_true.ndim == 2:
        # one-hot
        y_idx = y_true.argmax(axis=-1)
    else:
        y_idx = y_true
    return float(np.mean([y_idx[i] in top_k[i] for i in range(len(y_idx))]))


def confusion_matriz(y_true, y_pred, n_clases):
    """Calcula matriz de confusion NxN."""
    M = np.zeros((n_clases, n_clases), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


def f1_macro(cm):
    """F1 macro: promedio de F1 por clase."""
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def augmentar_flip_horizontal(img, semilla=0):
    """Volteo horizontal."""
    rng = np.random.default_rng(semilla)
    if rng.random() > 0.5:
        return img[:, ::-1].copy()
    return img


def augmentar_random_crop(img, pad=4, semilla=0):
    """Crop aleatorio con padding (para CIFAR-style)."""
    H, W = img.shape[:2]
    if img.ndim == 2:
        img_p = np.pad(img, pad, mode='reflect')
    else:
        img_p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    rng = np.random.default_rng(semilla)
    y0 = int(rng.integers(0, 2 * pad + 1))
    x0 = int(rng.integers(0, 2 * pad + 1))
    return img_p[y0:y0 + H, x0:x0 + W]


def main() -> int:
    # Ejemplo: 4 imagenes, 3 clases
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3], [0.1, 0.2, 3.0], [1.0, 1.0, 1.0]])
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    print(f"Cross-entropy: {cross_entropy(logits, y_true):.3f}")
    print(f"Top-1 accuracy: {(softmax(logits).argmax(-1) == y_true.argmax(-1)).mean():.3f}")
    print(f"Top-2 accuracy: {top_k_accuracy(logits, y_true, k=2):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())