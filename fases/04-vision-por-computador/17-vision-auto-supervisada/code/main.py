"""
Lección: 17-vision-auto-supervisada
Fase: 04
Prerrequisitos: 14-vision-transformers
"""
from __future__ import annotations
import sys
import numpy as np


def augmentation_doble(img, semilla=0):
    """Aplica dos augmentations aleatorias a la misma imagen.
    Devuelve dos 'vistas' que deberian ser similares en el espacio de features."""
    rng = np.random.default_rng(semilla)
    # Crop aleatorio
    H, W = img.shape[:2]
    crop = max(1, int(0.8 * min(H, W)))
    y0 = int(rng.integers(0, H - crop + 1))
    x0 = int(rng.integers(0, W - crop + 1))
    crop1 = img[y0:y0 + crop, x0:x0 + crop]
    # Resize (o padding) a tamano original
    crop1 = np.pad(crop1, ((0, H - crop), (0, W - crop), (0, 0)), mode='reflect')
    # Flip
    flip = rng.random() > 0.5
    crop2 = img[:, ::-1].copy() if flip else img.copy()
    return crop1, crop2


def simclr_loss(z1, z2, temperatura=0.1):
    """SimCLR contrastive loss: maximizar similitud entre pares positivos.
    z1, z2: embeddings L2-normalizados de shape (N, D)."""
    n = len(z1)
    z = np.vstack([z1, z2])
    sim = z @ z.T / temperatura
    # Mascara: pares positivos en diagonal offset n
    etiquetas = np.concatenate([np.arange(n), np.arange(n)])
    # Softmax
    exp = np.exp(sim - sim.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    # Loss: -log(prob[i, positivos])
    loss = 0.0
    for i in range(n):
        pos = i + n
        loss -= np.log(probs[i, pos] + 1e-10)
        loss -= np.log(probs[pos, i] + 1e-10)
    return loss / (2 * n)


def mae_mask(patches, ratio=0.75, semilla=0):
    """Masked Autoencoder: enmascara ratio% de patches.
    Devuelve patches_enmascarados, indices enmascarados, indices visibles."""
    rng = np.random.default_rng(semilla)
    n = len(patches)
    n_mask = int(n * ratio)
    indices = rng.permutation(n)
    masked_idx = indices[:n_mask]
    visible_idx = indices[n_mask:]
    return patches[visible_idx], masked_idx, visible_idx


def main() -> int:
    img = np.random.default_rng(0).normal(size=(32, 32, 3))
    v1, v2 = augmentation_doble(img, semilla=42)
    print(f"Vista 1: {v1.shape}, Vista 2: {v2.shape}")
    z1 = np.random.default_rng(0).normal(size=(8, 128))
    z1 = z1 / np.linalg.norm(z1, axis=-1, keepdims=True)
    z2 = np.random.default_rng(1).normal(size=(8, 128))
    z2 = z2 / np.linalg.norm(z2, axis=-1, keepdims=True)
    loss = simclr_loss(z1, z2, temperatura=0.1)
    print(f"SimCLR loss: {loss:.3f}")
    patches = np.random.default_rng(0).normal(size=(49, 768))  # 7x7
    vis, mid, vid = mae_mask(patches, ratio=0.75)
    print(f"Visible: {vis.shape}, masked idx: {len(mid)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())