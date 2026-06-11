"""
Lección: 05-stylegan
Fase: 08
StyleGAN: AdaIN (adaptive instance norm), mapping network z -> w, style mixing.
"""
from __future__ import annotations
import sys
import numpy as np


def affine_transform(z, W, b):
    """Mapping network: z -> w via affine.
    z: (n, latent_dim). W: (latent_dim, w_dim). b: (w_dim,).
    """
    return z @ W + b


def adain(x, w_y, w_b):
    """Adaptive Instance Normalization.
    x: (n, C) simplificado. w_y, w_b: (n, C).
    Normaliza x por instancia, escala y shift con w.
    """
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + 1e-6)
    return w_y * x_norm + w_b


def style_mod_conv(x, w):
    """Modular conv: peso * w (style). Pesos tienen shape (C_out, C_in, K, K).
    Simplificado: x (n, C_in) -> conv sin kernel real, aplica style scaling.
    """
    # Mock: w multiplica el output channel-wise
    return x * w.reshape(1, -1)


def noise_injection(x, noise, w_noise):
    """Add per-pixel noise scaled by w_noise. StyleGAN: w_noise learned per layer."""
    return x + w_noise * noise


def style_mixing(w1, w2, threshold, seed=0):
    """Style mixing: usar w1 para low res, w2 para high res.
    threshold: hasta que layer hacer crossover.
    """
    rng = np.random.default_rng(seed)
    layer = int(rng.integers(1, threshold)) if threshold > 1 else 1
    # Mezclar: rows [0, layer) de w1, [layer, end) de w2
    mixed = np.concatenate([w1[:layer], w2[layer:]], axis=0)
    return mixed, layer


def truncation_trick(w, psi, w_avg):
    """Truncation: w' = w_avg + psi * (w - w_avg).
    psi=0: siempre w_avg (avg face). psi=1: identidad.
    """
    return w_avg + psi * (w - w_avg)


def progressive_grow_layers(target_res, latent_dim=8, seed=0):
    """Mock: lista de layers a anadir para progressive growing.
    target_res: target resolution. Cada step dobla res.
    """
    layers = []
    rng = np.random.default_rng(seed)
    in_dim = latent_dim
    res = 4
    while res < target_res:
        layers.append(f"conv_{res}x{res}")
        in_dim *= 2
        res *= 2
    return layers


def main() -> int:
    rng = np.random.default_rng(0)
    z = rng.standard_normal((4, 8))
    # Mapping: z -> w
    W = rng.standard_normal((8, 16)) * 0.1
    b = np.zeros(16)
    w = affine_transform(z, W, b)
    print(f"z: {z.shape} -> w: {w.shape}")
    # AdaIN
    x = rng.standard_normal((4, 16))
    w_y = rng.standard_normal((4, 16)) * 0.1
    w_b = rng.standard_normal((4, 16)) * 0.1
    out = adain(x, w_y, w_b)
    print(f"AdaIN: {x.shape} -> {out.shape}, mean={out.mean():.3f}, std={out.std():.3f}")
    # Style mixing
    w2 = rng.standard_normal((4, 16))
    mixed, layer = style_mixing(w, w2, threshold=2, seed=0)
    print(f"Style mixing layer {layer}: shape {mixed.shape}")
    # Truncation
    w_avg = w.mean(axis=0)
    w_trunc = truncation_trick(w, psi=0.5, w_avg=w_avg)
    print(f"Truncation psi=0.5: shape {w_trunc.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())