"""
Lección: 18-clip-vocabulario-abierto
Fase: 04
Prerrequisitos: 17-vision-auto-supervisada
"""
from __future__ import annotations
import sys
import numpy as np


def text_encoder_mock(prompt, dim=512):
    """Mock: convierte texto en embedding deterministico (hash)."""
    rng = np.random.default_rng(hash(prompt) % 2**32)
    return rng.normal(0, 1, size=(dim,))


def image_encoder_mock(img, dim=512, semilla=0):
    """Mock: convierte imagen en embedding."""
    H, W, C = img.shape
    bloques = H // 8 * W // 8
    rng = np.random.default_rng(semilla + H + W)
    return rng.normal(0, 1, size=(dim,))


def l2_normalize(x):
    """Normaliza a norma 1."""
    n = np.linalg.norm(x)
    if n == 0:
        return x
    return x / n


def clip_similarity(img_emb, txt_emb):
    """Similitud coseno entre imagen y texto."""
    img = l2_normalize(img_emb)
    txt = l2_normalize(txt_emb)
    return float(img @ txt)


def zero_shot_classify(img_emb, etiquetas, dim=512):
    """Zero-shot: similitud entre imagen y cada clase (texto), devuelve la mas probable."""
    sims = []
    for label in etiquetas:
        txt_emb = text_encoder_mock(label, dim=dim)
        sims.append(clip_similarity(img_emb, txt_emb))
    mejor = int(np.argmax(sims))
    return mejor, sims


def main() -> int:
    img = np.random.default_rng(0).normal(size=(224, 224, 3))
    img_emb = image_encoder_mock(img, dim=512, semilla=42)
    print(f"Image emb: {img_emb.shape}, norm: {np.linalg.norm(img_emb):.3f}")
    # Zero-shot con clases
    clases = ["cat", "dog", "car", "tree"]
    pred, sims = zero_shot_classify(img_emb, clases)
    print(f"Clases: {clases}, sims: {[f'{s:.2f}' for s in sims]}")
    print(f"Prediccion: {clases[pred]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())