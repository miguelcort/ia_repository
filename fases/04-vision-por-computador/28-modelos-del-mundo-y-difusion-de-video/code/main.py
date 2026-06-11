"""
Lección: 28-modelos-del-mundo-y-difusion-de-video
Fase: 04
Prerrequisitos: 23-difusion-transformers-y-flujo-rectificado
"""
from __future__ import annotations
import sys
import numpy as np


def spacetime_patches(video, patch_t=2, patch_h=8, patch_w=8):
    """Divide video (T, H, W, C) en patches espacio-temporales.
    Output: (n_patches, patch_t * patch_h * patch_w * C)."""
    T, H, W, C = video.shape
    assert T % patch_t == 0
    n_t = T // patch_t
    n_h = H // patch_h
    n_w = W // patch_w
    patches = video.reshape(n_t, patch_t, n_h, patch_h, n_w, patch_w, C)
    patches = patches.transpose(0, 2, 4, 1, 3, 5, 6)  # n_t, n_h, n_w, p_t, p_h, p_w, C
    return patches.reshape(n_t * n_h * n_w, -1)


def posicion_3d_sin_cos(n_patches_t, n_patches_h, n_patches_w, dim, semilla=0):
    """Positional encoding 3D: senos/cosenos sobre t, h, w."""
    pe = np.zeros((n_patches_t * n_patches_h * n_patches_w, dim))
    for ti in range(n_patches_t):
        for hi in range(n_patches_h):
            for wi in range(n_patches_w):
                idx = (ti * n_patches_h + hi) * n_patches_w + wi
                for d in range(dim):
                    if d % 6 < 2:
                        pe[idx, d] = np.sin(ti / (10000 ** (d / dim)))
                    elif d % 6 < 4:
                        pe[idx, d] = np.sin(hi / (10000 ** (d / dim)))
                    else:
                        pe[idx, d] = np.sin(wi / (10000 ** (d / dim)))
    return pe


def world_model_step(state, accion, dim_accion, dim_estado, semilla=0):
    """Mock: world model simple. estado -> accion -> siguiente estado.
    En Ha et al. (Dreamer, DreamerV2), world model es un SSM/RSSM entrenado con reconstruccion + reward."""
    rng = np.random.default_rng(semilla)
    if accion.shape[0] != dim_accion:
        accion = accion[:dim_accion]
    # dinamica: x_t+1 = x_t + W * accion + ruido
    W = rng.normal(scale=0.1, size=(dim_estado, dim_accion))
    next_state = state + W @ accion
    return next_state


def main() -> int:
    video = np.random.default_rng(0).normal(size=(4, 16, 16, 3))
    patches = spacetime_patches(video, patch_t=2, patch_h=8, patch_w=8)
    print(f"Spacetime patches: {patches.shape}")
    pe = posicion_3d_sin_cos(2, 2, 2, dim=64)
    print(f"PE 3D: {pe.shape}")
    # World model
    state = np.zeros(8)
    accion = np.ones(3)
    next_state = world_model_step(state, accion, dim_accion=3, dim_estado=8)
    print(f"Next state: {next_state.round(3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())