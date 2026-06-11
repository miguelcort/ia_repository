"""
Lección: 01-vision-transformer-y-patch-tokens
Fase: 12
Vision Transformer (ViT): patches, linear projection, transformer encoder.
Patch tokens como input. Dosovitskiy 2020.
"""
from __future__ import annotations
import sys
import numpy as np


def image_to_patches(img, patch_size=16):
    """Image (H, W, C) -> patches (n_patches, patch_size^2 * C)."""
    H, W, C = img.shape
    n_h, n_w = H // patch_size, W // patch_size
    return img[:n_h * patch_size, :n_w * patch_size, :].reshape(
        n_h, patch_size, n_w, patch_size, C
    ).transpose(0, 2, 1, 3, 4).reshape(n_h * n_w, -1)


def patch_embedding(patches, W_proj, b_proj):
    """Linear projection de patches a d_model."""
    return patches @ W_proj + b_proj


def add_class_token(emb, cls_emb):
    """Prepend [CLS] token. emb: (n, d) -> (n+1, d)."""
    return np.vstack([cls_emb.reshape(1, -1), emb])


def add_positional_embedding(emb, pos_emb):
    """Add positional embedding (learned or sinusoidal)."""
    return emb + pos_emb[:emb.shape[0]]


def vit_forward(x, W_proj, b_proj, cls_emb, pos_emb):
    """x: (n_patches, patch_dim) -> (n_patches + 1, d_model)."""
    emb = patch_embedding(x, W_proj, b_proj)
    emb = add_class_token(emb, cls_emb)
    emb = add_positional_embedding(emb, pos_emb)
    return emb


def n_patches(image_size=224, patch_size=16):
    """Numero de patches para ViT."""
    return (image_size // patch_size) ** 2


def main() -> int:
    rng = np.random.default_rng(0)
    img = rng.standard_normal((224, 224, 3))
    patches = image_to_patches(img, patch_size=16)
    print(f"Image 224x224x3 -> {patches.shape[0]} patches de {patches.shape[1]} dim")
    n = n_patches(224, 16)
    print(f"+ 1 CLS token = {n + 1} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())