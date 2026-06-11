"""
Lección: 24-sam3-segmentacion-de-vocabulario-abierto
Fase: 04
Prerrequisitos: 18-clip-vocabulario-abierto
"""
from __future__ import annotations
import sys
import numpy as np


def sam_image_encoder_mock(img, dim=256, semilla=0):
    """Mock: ViT image encoder de SAM."""
    H, W = img.shape[:2]
    rng = np.random.default_rng(semilla + H + W)
    return rng.normal(0, 1, size=(H, W, dim))


def sam_prompt_encoder_point(points, labels):
    """Codifica puntos (x, y) y labels (1=foreground, 0=background)
    como embeddings posicionales."""
    if not points:
        return np.zeros((0, 256))
    pos = np.array(points, dtype=np.float32)
    pos /= 1024.0  # normalizar a [-1, 1] approx
    pos -= 0.5
    pos *= 2  # [-1, 1]
    lbl = np.array(labels, dtype=np.float32).reshape(-1, 1)
    # Combinar: pos (2 dims) + label (1 dim) -> 3 dims, proyectar a 256
    feats = np.concatenate([pos, lbl], axis=-1)
    W = np.random.default_rng(0).normal(scale=0.5, size=(3, 256))
    return feats @ W


def sam_mask_decoder_mock(img_emb, prompt_emb):
    """Mock: combina image embedding + prompt embeddings -> mascara binaria.
    En SAM real, usa cross-attention y transposed conv."""
    # Promedio y proyeccion simple
    if len(prompt_emb) == 0:
        return np.zeros(img_emb.shape[:2], dtype=bool)
    p = prompt_emb.mean(axis=0)
    # Producto punto con canales de img_emb
    sim = (img_emb * p).sum(axis=-1)
    threshold = sim.mean()
    return sim > threshold


def sam_text_prompt_embedding(text, dim=256):
    """SAM3 anade encoding de texto para open-vocab segmentation."""
    rng = np.random.default_rng(hash(text) % 2**32)
    return rng.normal(0, 1, size=(dim,))


def iou_mask(pred, gt):
    """IoU entre mascaras binarias."""
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def main() -> int:
    img = np.random.default_rng(0).normal(size=(64, 64, 3))
    img_emb = sam_image_encoder_mock(img, dim=256)
    # Prompt: 1 punto foreground
    prompts = sam_prompt_encoder_point([(32, 32)], [1])
    print(f"Prompt embedding: {prompts.shape}")
    mask = sam_mask_decoder_mock(img_emb, prompts)
    print(f"Mask shape: {mask.shape}, area: {mask.sum()}")
    # Open-vocab: prompt textual
    text_emb = sam_text_prompt_embedding("cat")
    print(f"Text embedding: {text_emb.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())