"""
Lección: 05-transfer-learning
Fase: 04
Prerrequisitos: 04-clasificacion-de-imagenes
"""
from __future__ import annotations
import sys
import numpy as np


def simulador_extraccion_features(x):
    """Simula pasar imagenes por un modelo preentrenado (e.g. ResNet50).
    Devuelve features de dimension fija."""
    # Promedio por bloques de 16 pixeles
    H, W, C = x.shape
    bloques_y = H // 4
    bloques_x = W // 4
    features = np.zeros((1, bloques_y * bloques_x * C))
    idx = 0
    for i in range(bloques_y):
        for j in range(bloques_x):
            bloque = x[i*4:(i+1)*4, j*4:(j+1)*4, :]
            features[0, idx] = bloque.mean()
            idx += 1
    return features


def simulador_cabeza_clasificacion(features, n_clases=10, semilla=0):
    """Simula el FC final: features (N, D) -> (N, n_clases)."""
    rng = np.random.default_rng(semilla)
    N, D = features.shape
    W = rng.normal(scale=0.01, size=(D, n_clases))
    b = np.zeros(n_clases)
    return features @ W + b


def estrategia_lr_diferencial(base_lr, ratio_backbone=0.1):
    """Learning rate mas bajo para backbone preentrenado, mas alto para cabeza nueva."""
    return {
        "backbone": base_lr * ratio_backbone,
        "cabeza": base_lr,
    }


def congelar_capas(n_total, n_congeladas):
    """Devuelve una mascara booleana: True = entrenar, False = congelar."""
    return np.array([False] * n_congeladas + [True] * (n_total - n_congeladas))


def main() -> int:
    img = np.random.default_rng(0).normal(size=(32, 32, 3))
    features = simulador_extraccion_features(img)
    print(f"Features shape: {features.shape}")
    logits = simulador_cabeza_clasificacion(features, n_clases=10)
    print(f"Logits shape: {logits.shape}")
    lrs = estrategia_lr_diferencial(1e-3, 0.1)
    print(f"LRs: backbone={lrs['backbone']}, cabeza={lrs['cabeza']}")
    mask = congelar_capas(10, 7)
    print(f"Mascara congelamiento: {mask}")
    return 0


if __name__ == "__main__":
    sys.exit(main())