"""
Lección: 09-vision-transformers
Fase: 07
ViT: divide imagen en patches, los proyecta a embeddings, transformer encoder.
"""
from __future__ import annotations
import sys
import numpy as np


def image_to_patches(img, patch_size):
    """img: (H, W, C). Returns: (n_patches, patch_size^2 * C)."""
    H, W, C = img.shape
    assert H % patch_size == 0 and W % patch_size == 0
    n_h, n_w = H // patch_size, W // patch_size
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patch = img[i*patch_size:(i+1)*patch_size,
                        j*patch_size:(j+1)*patch_size, :]
            patches.append(patch.flatten())
    return np.stack(patches)


def patch_embeddings(patches, W_patch, b_patch):
    """Linear projection: (n, patch_dim) -> (n, d_model)."""
    return patches @ W_patch.T + b_patch


def add_class_token(emb, cls_emb):
    """Prepend [CLS] token. emb: (n, d)."""
    n, d = emb.shape
    out = np.zeros((n + 1, d))
    out[0] = cls_emb
    out[1:] = emb
    return out


def add_2d_positional(emb, n_patches_h, n_patches_w, pe_2d):
    """Suma PE 2D aprendido (precomputado)."""
    cls_pe = pe_2d[0]
    spatial_pe = pe_2d[1:].reshape(n_patches_h, n_patches_w, -1)
    pe_seq = np.zeros((emb.shape[0], emb.shape[1]))
    pe_seq[0] = cls_pe
    idx = 1
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            pe_seq[idx] = spatial_pe[i, j]
            idx += 1
    return emb + pe_seq


def vit_classifier(x, W_head):
    """Usa [CLS] (pos 0) para clasificar. x: (n+1, d)."""
    return x[0] @ W_head.T


def main() -> int:
    img = np.random.default_rng(0).standard_normal((32, 32, 3))
    patch_size = 8
    patches = image_to_patches(img, patch_size)
    print(f"Imagen {img.shape} -> {patches.shape[0]} patches de {patches.shape[1]}")
    d_model = 64
    W_patch = np.random.default_rng(1).standard_normal((d_model, patches.shape[1])) * 0.02
    b_patch = np.zeros(d_model)
    emb = patch_embeddings(patches, W_patch, b_patch)
    cls_emb = np.zeros(d_model)
    full = add_class_token(emb, cls_emb)
    print(f"Con [CLS]: {full.shape}")
    pe_2d = np.random.default_rng(2).standard_normal((full.shape[0], d_model)) * 0.02
    full_pe = add_2d_positional(full, 4, 4, pe_2d)
    W_head = np.random.default_rng(3).standard_normal((10, d_model)) * 0.02
    logits = vit_classifier(full_pe, W_head)
    print(f"Logits shape: {logits.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())