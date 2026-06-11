"""
Lección: 08-controlnet-y-lora-condicionamiento
Fase: 08
ControlNet: spatial conditioning (edges, depth, pose). LoRA: low-rank adaptacion.
"""
from __future__ import annotations
import sys
import numpy as np


def zero_convolution(W, b, seed=0):
    """Zero convolution: 1x1 conv con pesos inicializados en 0.
    Empieza con output 0, gradualmente aprende.
    """
    rng = np.random.default_rng(seed)
    W_z = np.zeros_like(W) if W.ndim == 2 else rng.standard_normal(W.shape) * 1e-4
    b_z = np.zeros_like(b) if b is not None else None
    return W_z, b_z


def controlnet_block(x, ctrl, ctrl_W, ctrl_b, base_out):
    """ControlNet: y = base_out + ctrl @ ctrl_W (zero conv).
    Inicialmente ctrl @ ctrl_W = 0, identidad.
    """
    if x.shape == ctrl.shape:
        # Aplicar zero conv al control
        ctrl_out = ctrl @ ctrl_W
        return base_out + ctrl_out
    return base_out


def controlnet_zero_init(W, scale=1e-4):
    """Init para zero conv weights: cercanos a 0."""
    return np.full_like(W, scale * np.random.default_rng(0).standard_normal())


def lora_init(in_dim, out_dim, rank=4, seed=0):
    """LoRA: A in (in, rank), B in (rank, out). Init A=random, B=0 -> BA=0."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((in_dim, rank))
    B = np.zeros((rank, out_dim))
    return A, B


def lora_forward(x, W, A, B, alpha=1.0):
    """Forward: y = x @ W + (alpha / rank) * x @ A @ B."""
    rank = A.shape[1]
    return x @ W + (alpha / rank) * (x @ A @ B)


def merge_lora(W, A, B, alpha=1.0):
    """Merge LoRA into W: W_new = W + (alpha/rank) * A @ B.
    En inference: W_new reemplaza W, no overhead.
    """
    rank = A.shape[1]
    return W + (alpha / rank) * (A @ B)


def controlnet_conditions():
    """Tipos de conditioning para ControlNet."""
    return {
        "Canny edges": "Edge detection (Canny)",
        "Depth": "MiDaS, DPT, ZoeDepth",
        "Pose": "OpenPose, DWPose",
        "Segmentation": "SegFormer, Mask2Former",
        "Normal": "Normal map estimation",
        "Inpaint": "Mask + image",
        "IP-Adapter": "Image prompt (CLIP image emb)",
    }


def main() -> int:
    # LoRA
    in_dim, out_dim, rank = 8, 16, 2
    rng = np.random.default_rng(0)
    W = rng.standard_normal((in_dim, out_dim)) * 0.1
    A, B = lora_init(in_dim, out_dim, rank=rank, seed=0)
    x = rng.standard_normal((3, in_dim))
    # Inicialmente BA = 0
    out_init = lora_forward(x, W, A, B)
    print(f"LoRA init: x @ A @ B deberia ser 0: {np.allclose(x @ A @ B, 0)}")
    # Merge
    W_merged = merge_lora(W, A, B)
    out_merged = x @ W_merged
    print(f"LoRA merged: forward match: {np.allclose(out_init, out_merged)}")
    # ControlNet conditions
    print("\nControlNet conditions:")
    for k, v in controlnet_conditions().items():
        print(f"  {k:15s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())