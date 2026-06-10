"""
Lección: 04-funciones-de-activacion
Fase: 03
Prerrequisitos: 02-redes-multicapa
"""
from __future__ import annotations
import sys
import numpy as np


def sigmoid(z):
    """1 / (1 + e^-z). Rango (0, 1). Satura en ambos lados."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivada(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    """Rango (-1, 1). Centrada en cero."""
    return np.tanh(z)


def tanh_derivada(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    """max(0, z). No satura en positivo, gradiente 0 en negativo (dying ReLU)."""
    return np.maximum(0, z)


def relu_derivada(z):
    return (z > 0).astype(float)


def leaky_relu(z, alpha=0.01):
    """max(alpha*z, z). Variante que no muere en negativo."""
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivada(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


def gelu(z):
    """Gaussian Error Linear Unit. Usada en transformers (BERT, GPT)."""
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))


def gelu_derivada(z):
    # Aproximacion numerica estable
    h = 1e-4
    return (gelu(z + h) - gelu(z - h)) / (2 * h)


def softmax(z):
    """Convierte logits en probabilidades. Resta max por estabilidad."""
    z_est = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z_est)
    return exp / exp.sum(axis=-1, keepdims=True)


def main() -> int:
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    print(f"sigmoid(-2..2) = {sigmoid(z)}")
    print(f"tanh(-2..2)    = {tanh(z)}")
    print(f"relu(-2..2)    = {relu(z)}")
    print(f"leaky_relu(-2..2) = {leaky_relu(z)}")
    print(f"gelu(-2..2)    = {gelu(z).round(3)}")
    logits = np.array([1.0, 2.0, 3.0])
    print(f"softmax([1,2,3]) = {softmax(logits)} (suma={softmax(logits).sum()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())