"""
Lección: 09-generacion-de-imagenes-gan
Fase: 04
Prerrequisitos: 03-cnns-desde-lenet-hasta-resnet
"""
from __future__ import annotations
import sys
import numpy as np


def ruido(latente_dim, n_muestras, semilla=0):
    """Vector latente z ~ N(0, 1). Input del generador."""
    rng = np.random.default_rng(semilla)
    return rng.normal(0, 1, size=(n_muestras, latente_dim))


def leaky_relu(z, alpha=0.2):
    return np.where(z > 0, z, alpha * z)


def generador_minimo(z, salida_dim=784):
    """Generador simple: 100 -> 256 -> 784 (imagen 28x28)."""
    rng = np.random.default_rng(0)
    W1 = rng.normal(scale=0.1, size=(z.shape[1], 256))
    W2 = rng.normal(scale=0.1, size=(256, salida_dim))
    h = leaky_relu(z @ W1)
    out = np.tanh(h @ W2)
    return out


def discriminador_minimo(x, semilla=0):
    """Discriminador: 784 -> 256 -> 1 (real/fake)."""
    rng = np.random.default_rng(semilla)
    W1 = rng.normal(scale=0.1, size=(x.shape[1], 256))
    W2 = rng.normal(scale=0.1, size=(256, 1))
    h = leaky_relu(x @ W1)
    out = h @ W2
    return out  # logits


def bce_logits(y_true, logits, eps=1e-9):
    """BCE sobre logits (numericamente estable)."""
    return float(np.mean(
        np.maximum(logits, 0) - logits * y_true + np.log(1 + np.exp(-np.abs(logits)))
    ))


def grad_discriminador(x_real, x_fake, semilla=0):
    """Calcula gradiente del discriminador respecto a sus pesos (aproximado).
    Para simplificacion devolvemos solo el cambio en la salida promedio."""
    d_real = discriminador_minimo(x_real, semilla)
    d_fake = discriminador_minimo(x_fake, semilla)
    # D quiere d_real grande, d_fake pequeno
    loss_d = bce_logits(np.ones_like(d_real), d_real) + bce_logits(np.zeros_like(d_fake), d_fake)
    return loss_d


def grad_generador(z, semilla=0):
    """Para el generador, queremos que D clasifique fake como real."""
    x_fake = generador_minimo(z)
    d_fake = discriminador_minimo(x_fake, semilla)
    loss_g = bce_logits(np.ones_like(d_fake), d_fake)
    return loss_g


def main() -> int:
    z = ruido(100, 16, semilla=42)
    x_fake = generador_minimo(z)
    x_real = np.random.default_rng(0).normal(0, 1, size=(16, 784)) * 0.5
    print(f"z shape: {z.shape}, x_fake: {x_fake.shape}")
    loss_d = grad_discriminador(x_real, x_fake)
    loss_g = grad_generador(z)
    print(f"Loss D: {loss_d:.3f}, Loss G: {loss_g:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())