"""
Lección: 07-regularizacion
Fase: 03
Prerrequisitos: 02-redes-multicapa
"""
from __future__ import annotations
import sys
import numpy as np


def dropout_forward(x, p=0.5, entrenamiento=True, semilla=0):
    """Inactivar aleatoriamente el p% de las neuronas durante entrenamiento.
    En inference no se aplica (solo escalado)."""
    if not entrenamiento or p == 0:
        return x, None
    rng = np.random.default_rng(semilla)
    mask = rng.binomial(1, 1 - p, size=x.shape)
    # Inverted dropout: escala durante entrenamiento
    return x * mask / (1 - p), mask


def dropout_backward(grad, mask, p=0.5):
    """Aplica la misma mascara al gradiente."""
    if mask is None:
        return grad
    return grad * mask / (1 - p)


def l2_pesos(W, lambda_=0.01):
    """Loss L2: lambda * sum(W^2). Suaviza pesos."""
    return lambda_ * np.sum(W ** 2)


def l2_grad(W, lambda_=0.01):
    """Gradiente de L2: 2 * lambda * W."""
    return 2 * lambda_ * W


def l1_pesos(W, lambda_=0.01):
    """Loss L1: lambda * sum(|W|). Promueve sparsity (w=0)."""
    return lambda_ * np.sum(np.abs(W))


def l1_grad(W, lambda_=0.01):
    """Gradiente de L1: lambda * sign(W)."""
    return lambda_ * np.sign(W)


def batch_norm_forward(x, gamma=1.0, beta=0.0, eps=1e-5, entrenamiento=True,
                       media_movil=None, var_movil=None, momentum=0.9):
    """Batch Normalization: normaliza x por minibatch en entrenamiento, por estadisticas moviles en inference."""
    if entrenamiento:
        mu = x.mean(axis=0)
        var = x.var(axis=0)
        if media_movil is not None:
            media_movil[:] = momentum * media_movil + (1 - momentum) * mu
            var_movil[:] = momentum * var_movil + (1 - momentum) * var
        x_norm = (x - mu) / np.sqrt(var + eps)
    else:
        if media_movil is None or var_movil is None:
            mu = x.mean(axis=0)
            var = x.var(axis=0)
        else:
            mu = media_movil
            var = var_movil
        x_norm = (x - mu) / np.sqrt(var + eps)
    return gamma * x_norm + beta, x_norm, mu, var


def early_stopping(historial_val, paciencia=3):
    """Para si el val loss no mejora en 'paciencia' epocas."""
    if len(historial_val) <= paciencia:
        return False
    mejor = min(historial_val[:-paciencia])
    ultimos = historial_val[-paciencia:]
    return all(v >= mejor for v in ultimos)


def main() -> int:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5))
    out, mask = dropout_forward(x, p=0.5, semilla=42)
    print(f"Dropout: inactivos={int((1-mask).sum() if mask is not None else 0)} de {x.size}")
    # L2/L1
    W = np.array([[0.1, 0.5], [-0.3, 0.8]])
    print(f"L2(W): {l2_pesos(W, 0.1):.3f}")
    print(f"L1(W): {l1_pesos(W, 0.1):.3f}")
    # BN
    out, x_norm, mu, var = batch_norm_forward(x)
    print(f"BN media: {out.mean(axis=0).round(3)}")
    print(f"BN std: {out.std(axis=0).round(3)}")
    # Early stopping
    print(f"Early stop?: {early_stopping([0.5, 0.4, 0.45, 0.46, 0.47], paciencia=3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())