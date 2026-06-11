"""
Lección: 15-visual-autoregressive-var
Fase: 08
VAR: Visual Autoregressive. Predice siguiente-scale en vez de next-token.
Nuevo paradigma: next-scale prediction.
"""
from __future__ import annotations
import sys
import numpy as np


def patchify_to_tokens(image, patch_size):
    """Image (H, W, C) -> tokens (n_patches, patch_size^2 * C)."""
    H, W, C = image.shape
    n_h, n_w = H // patch_size, W // patch_size
    return image[:n_h * patch_size, :n_w * patch_size, :].reshape(
        n_h, patch_size, n_w, patch_size, C
    ).transpose(0, 2, 1, 3, 4).reshape(n_h * n_w, -1)


def var_quantize(continuous_tokens, codebook, k=1):
    """Cuantizar continuous tokens a nearest codebook entry."""
    # k es numero de nearest neighbors (default 1 = argmin)
    flat = continuous_tokens.reshape(-1, continuous_tokens.shape[-1])
    dists = np.linalg.norm(flat[:, None] - codebook[None], axis=-1)
    idx = np.argmin(dists, axis=-1)
    return idx.reshape(continuous_tokens.shape[:-1])


def var_predict_scale(model_fn, prev_scales, scale_idx, codebook_size):
    """Predict next scale dado previous scales.
    model_fn: (prev_scales_flat, scale_idx) -> logits (n_tokens, codebook_size).
    Returns: predicted tokens.
    """
    flat = np.concatenate([s.flatten() for s in prev_scales])
    logits = model_fn(flat, scale_idx)
    return np.argmax(logits, axis=-1)


def next_scale_prediction(generated_scales, n_scales, target_res):
    """Genera la siguiente escala, dobla res cada step.
    Scales: 1x1, 2x2, 4x4, ..., target_res x target_res.
    """
    if len(generated_scales) == 0:
        return np.array([[0]])
    last = generated_scales[-1]
    # next res: dobla (mock)
    next_res = last.shape[0] * 2
    if next_res > target_res:
        return last
    return np.zeros((next_res, next_res))


def var_vs_diffusion():
    """Comparacion."""
    return {
        "Diffusion (DDPM)": "Pixel/feature space denoising, iterative",
        "AR (GPT-style)": "Token-by-token, left-to-right raster order",
        "VAR (next-scale)": "Multi-scale, coarse-to-fine",
        "Speed": "VAR 20x faster than AR, 10x faster than DiT",
        "Quality": "VAR comparable a diffusion en benchmarks",
        "VAR paper": "Visual Autoregressive Modeling (NeurIPS 2024)",
    }


def main() -> int:
    print("=== VAR vs diffusion ===")
    for k, v in var_vs_diffusion().items():
        print(f"  {k:20s} {v}")
    # Demo
    image = np.random.default_rng(0).standard_normal((8, 8, 3))
    tokens = patchify_to_tokens(image, patch_size=2)
    print(f"\n8x8x3 -> {tokens.shape[0]} tokens de {tokens.shape[1]} dim")
    return 0


if __name__ == "__main__":
    sys.exit(main())