"""
Lección: 01-fundamentos-de-imagen
Fase: 04
Prerrequisitos: 03-nucleo-deep-learning/10-mini-framework
"""
from __future__ import annotations
import sys
import numpy as np


def crear_imagen_rayas(alto=8, ancho=8, color=255):
    """Imagen basica: patron de rayas verticales."""
    img = np.zeros((alto, ancho), dtype=np.uint8)
    for i in range(ancho):
        if i % 2 == 0:
            img[:, i] = color
    return img


def crear_imagen_color(alto=8, ancho=8):
    """Imagen RGB: rojo en (0,0), verde en (1,0), etc."""
    img = np.zeros((alto, ancho, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]
    img[0, 1] = [0, 255, 0]
    img[0, 2] = [0, 0, 255]
    img[1, 0] = [255, 255, 0]
    img[1, 1] = [0, 255, 255]
    img[1, 2] = [255, 0, 255]
    return img


def rgb_a_grises(img):
    """Luminancia: 0.299*R + 0.587*G + 0.114*B."""
    if img.ndim == 2:
        return img
    pesos = np.array([0.299, 0.587, 0.114])
    return (img * pesos).sum(axis=-1).astype(img.dtype)


def normalizar_imagen(img, modo="01"):
    """Normaliza uint8 [0,255] a [0,1] o [-1,1]."""
    f = img.astype(np.float32) / 255.0
    if modo == "11":
        f = f * 2.0 - 1.0
    return f


def redimensionar_bilineal(img, nuevo_alto, nuevo_ancho):
    """Redimensiona imagen por interpolacion bilineal."""
    alto, ancho = img.shape[:2]
    if img.ndim == 3:
        canales = img.shape[2]
    else:
        canales = 1
    y_idx = np.linspace(0, alto - 1, nuevo_alto)
    x_idx = np.linspace(0, ancho - 1, nuevo_ancho)
    y0 = np.floor(y_idx).astype(int)
    y1 = np.minimum(y0 + 1, alto - 1)
    x0 = np.floor(x_idx).astype(int)
    x1 = np.minimum(x0 + 1, ancho - 1)
    wy = y_idx - y0
    wx = x_idx - x0
    out = np.zeros((nuevo_alto, nuevo_ancho, canales) if canales > 1 else (nuevo_alto, nuevo_ancho), dtype=img.dtype)
    for i, yi in enumerate(y_idx):
        for j, xj in enumerate(x_idx):
            y0i, y1i = y0[i], y1[i]
            x0j, x1j = x0[j], x1[j]
            wyi, wxj = wy[i], wx[j]
            v00 = img[y0i, x0j]
            v01 = img[y0i, x1j]
            v10 = img[y1i, x0j]
            v11 = img[y1i, x1j]
            out[i if canales == 1 else i, j if canales == 1 else j] = (
                v00 * (1 - wyi) * (1 - wxj)
                + v01 * (1 - wyi) * wxj
                + v10 * wyi * (1 - wxj)
                + v11 * wyi * wxj
            )
    return out


def main() -> int:
    img_rayas = crear_imagen_rayas(8, 8)
    print(f"Imagen rayas shape: {img_rayas.shape}, dtype: {img_rayas.dtype}")
    img_color = crear_imagen_color(2, 3)
    print(f"Imagen color shape: {img_color.shape}")
    img_gris = rgb_a_grises(img_color)
    print(f"Imagen gris shape: {img_gris.shape}")
    img_norm = normalizar_imagen(img_rayas, modo="01")
    print(f"Imagen normalizada rango: [{img_norm.min():.2f}, {img_norm.max():.2f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())