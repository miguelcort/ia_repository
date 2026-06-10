"""
Lección: 02-convoluciones-desde-cero
Fase: 04
Prerrequisitos: 01-fundamentos-de-imagen
"""
from __future__ import annotations
import sys
import numpy as np


def conv2d(x, kernel, padding=0, stride=1):
    """Convolucion 2D. x: (H, W) o (H, W, C_in). kernel: (kH, kW, C_in, C_out)."""
    if x.ndim == 2:
        x = x[:, :, None]
    if kernel.ndim == 2:
        kernel = kernel[:, :, None, None]
    H, W, C_in = x.shape
    kH, kW, _, C_out = kernel.shape
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding), (0, 0)))
        H, W = x.shape[:2]
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    salida = np.zeros((out_H, out_W, C_out))
    for i in range(out_H):
        for j in range(out_W):
            patch = x[i * stride:i * stride + kH, j * stride:j * stride + kW, :]
            for c in range(C_out):
                salida[i, j, c] = (patch * kernel[:, :, :, c]).sum()
    if C_out == 1:
        return salida[:, :, 0]
    return salida


def kernel_borde_vertical():
    """Detector de bordes verticales de Sobel."""
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)


def kernel_borde_horizontal():
    """Detector de bordes horizontales de Sobel."""
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


def kernel_blur():
    """Box blur 3x3."""
    return np.ones((3, 3)) / 9.0


def kernel_nitidez():
    """Filtro de nitidez (sharpen)."""
    return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=float)


def max_pool2d(x, size=2, stride=2):
    """Max pooling 2D. x: (H, W) o (H, W, C)."""
    una_capa = x.ndim == 2
    if una_capa:
        x = x[:, :, None]
    H, W, C = x.shape
    out_H = (H - size) // stride + 1
    out_W = (W - size) // stride + 1
    salida = np.zeros((out_H, out_W, C))
    for i in range(out_H):
        for j in range(out_W):
            patch = x[i * stride:i * stride + size, j * stride:j * stride + size, :]
            salida[i, j, :] = patch.max(axis=(0, 1))
    if una_capa:
        return salida[:, :, 0]
    return salida


def avg_pool2d(x, size=2, stride=2):
    """Average pooling 2D."""
    una_capa = x.ndim == 2
    if una_capa:
        x = x[:, :, None]
    H, W, C = x.shape
    out_H = (H - size) // stride + 1
    out_W = (W - size) // stride + 1
    salida = np.zeros((out_H, out_W, C))
    for i in range(out_H):
        for j in range(out_W):
            patch = x[i * stride:i * stride + size, j * stride:j * stride + size, :]
            salida[i, j, :] = patch.mean(axis=(0, 1))
    if una_capa:
        return salida[:, :, 0]
    return salida


def main() -> int:
    # Imagen de ejemplo: un borde vertical en el centro
    img = np.zeros((10, 10), dtype=float)
    img[:, 5:] = 1.0
    # Borde vertical
    borde = conv2d(img, kernel_borde_vertical())
    print(f"Borde vertical: max={borde.max():.2f}, min={borde.min():.2f}")
    # Blur
    blur = conv2d(img, kernel_blur())
    print(f"Blur: max={blur.max():.2f}, min={blur.min():.2f}")
    # Pool
    pooled = max_pool2d(img, size=2, stride=2)
    print(f"MaxPool: shape {pooled.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())