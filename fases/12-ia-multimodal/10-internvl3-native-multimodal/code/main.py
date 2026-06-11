"""
Lección: 10-internvl3-native-multimodal
Fase: 12
InternVL3 (Shanghai AI Lab 2024): native multimodal pretraining.
InternViT + Qwen2.5 LLM. Dynamic resolution. 1B-72B.
"""
from __future__ import annotations
import numpy as np


def intern_vit_forward(image, patch_size=14, embed_dim=1024):
    """InternViT-300M/6B forward: image -> (n_patches + 1, embed_dim)."""
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    H, W, C = image.shape
    n = (H // patch_size) * (W // patch_size) + 1
    return rng.standard_normal((n, embed_dim)) * 0.1


def intern_dynamic_tile(image, max_tiles=12, min_tiles=1):
    """InternVL3 dynamic tiling: hasta 12 tiles, aspect preservado."""
    H, W, C = image.shape
    aspect = W / H
    best = (1, 1)
    best_score = float("inf")
    for n in range(min_tiles, max_tiles + 1):
        for nh in range(1, n + 1):
            nw = n // nh
            if nh * nw != n:
                continue
            score = abs(np.log((nw / nh) / aspect))
            if score < best_score:
                best = (nh, nw)
                best_score = score
    nh, nw = best
    th, tw = H // nh, W // nw
    cropped = image[: th * nh, : tw * nw, :]
    return cropped.reshape(nh, th, nw, tw, C).transpose(0, 2, 1, 3, 4).reshape(nh * nw, th, tw, C)


def intern_mlp_projector(image_features, mlp_layers, target_dim=4096):
    """MLP projection: InternViT dim -> LLM dim."""
    h = image_features
    for W, b in mlp_layers:
        h = h @ W + b
        h = np.maximum(h, 0)  # GELU approx
    return h


def pixel_shuffle_reshape(features, scale=2):
    """Pixel shuffle: spatial downsample, channel upsample. (n, d) -> (n/scale^2, d*scale^2)."""
    n, d = features.shape
    n_new = n // (scale * scale)
    d_new = d * (scale * scale)
    rng = np.random.default_rng(0)
    # simulate reshape
    reshaped = features[: n_new * scale * scale].reshape(n_new, scale * scale, d)
    return reshaped.reshape(n_new, d * scale * scale)[:, :d_new]


def internvl3_forward(image, mlp_layers, target_dim=4096, max_tiles=12):
    """Image -> tiles -> InternViT -> pixel shuffle -> MLP -> LLM dim."""
    tiles = intern_dynamic_tile(image, max_tiles=max_tiles)
    all_feats = []
    for t in tiles:
        feats = intern_vit_forward(t, embed_dim=1024)
        feats = pixel_shuffle_reshape(feats, scale=2)
        all_feats.append(feats)
    flat = np.concatenate(all_feats, axis=0)
    return intern_mlp_projector(flat, mlp_layers, target_dim=target_dim)


def main() -> int:
    rng = np.random.default_rng(0)
    img = rng.standard_normal((448, 672, 3))
    tiles = intern_dynamic_tile(img, max_tiles=4)
    print(f"InternVL3 tiles: {tiles.shape}")
    # MLP con 2 capas
    mlp = [(rng.standard_normal((1024 * 4, 1024)) * 0.02, np.zeros(1024)),
           (rng.standard_normal((1024, 4096)) * 0.02, np.zeros(4096))]
    out = internvl3_forward(img, mlp, target_dim=4096, max_tiles=4)
    print(f"InternVL3 output: {out.shape}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())