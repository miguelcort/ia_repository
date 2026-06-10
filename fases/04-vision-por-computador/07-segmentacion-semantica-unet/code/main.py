"""
Lección: 07-segmentacion-semantica-unet
Fase: 04
Prerrequisitos: 03-cnns-desde-lenet-hasta-resnet
"""
from __future__ import annotations
import sys
import numpy as np


def downsample(x, factor=2):
    """Reduce HxW a la mitad."""
    H, W, C = x.shape
    return x.reshape(H // factor, factor, W // factor, factor, C).mean(axis=(1, 3))


def upsample(x, factor=2):
    """Aumenta HxW factor veces (nearest neighbor)."""
    H, W, C = x.shape
    out = np.repeat(np.repeat(x, factor, axis=0), factor, axis=1)
    return out


def dice_loss(y_true, y_pred, eps=1e-7):
    """Dice loss para segmentacion.
    D = 2 * |A ∩ B| / (|A| + |B|).
    loss = 1 - D."""
    intersection = (y_true * y_pred).sum()
    return 1 - (2 * intersection + eps) / (y_true.sum() + y_pred.sum() + eps)


def iou_segmentation(y_true, y_pred, eps=1e-7):
    """IoU binario pixel a pixel."""
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return float((intersection + eps) / (union + eps))


def unet_skip_connection(encoder_feat, decoder_feat):
    """Concatena features del encoder y decoder a lo largo de canales.
    En U-Net se hace crop o padding para igualar HxW."""
    # Asumimos mismo HxW
    return np.concatenate([encoder_feat, decoder_feat], axis=-1)


def main() -> int:
    x = np.arange(64, dtype=float).reshape(8, 8, 1)
    down = downsample(x, factor=2)
    print(f"Down: {down.shape}")  # (4, 4, 1)
    up = upsample(down, factor=2)
    print(f"Up: {up.shape}")  # (8, 8, 1)
    # Dice loss ejemplo
    y_true = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    y_pred = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    print(f"Dice loss: {dice_loss(y_true, y_pred):.3f}")
    print(f"IoU: {iou_segmentation(y_true, y_pred):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())