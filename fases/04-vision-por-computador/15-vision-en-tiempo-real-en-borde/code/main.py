"""
Lección: 15-vision-en-tiempo-real-en-borde
Fase: 04
Prerrequisitos: 14-vision-transformers
"""
from __future__ import annotations
import sys
import numpy as np


def cuantizar_int8(pesos_float32):
    """Cuantizacion post-entrenamiento a int8.
    Escala = max(|w|) / 127. Devuelve (q_weights, scale, zero_point)."""
    escala_max = float(np.abs(pesos_float32).max())
    if escala_max == 0:
        return np.zeros_like(pesos_float32, dtype=np.int8), 0.0, 0
    scale = escala_max / 127.0
    q = np.clip(np.round(pesos_float32 / scale), -128, 127).astype(np.int8)
    return q, scale, 0


def dequantizar(q, scale, zero_point=0):
    """Vuelve de int8 a float32."""
    return (q.astype(np.float32) - zero_point) * scale


def macs_conv2d(H, W, C_in, C_out, kH, kW, H_out, W_out):
    """Multiplica-acumula (operaciones) en una conv2d. Para estimar FLOPs."""
    return H_out * W_out * C_in * C_out * kH * kW


def params_conv2d(C_in, C_out, kH, kW):
    """Parametros de una conv2d."""
    return C_in * C_out * kH * kW + C_out


def flops_yolo_nano(input_size=640):
    """FLOPs estimados de YOLOv8-Nano (mock: numero fijo).
    YOLOv8-N: ~8.7 GFLOPs."""
    return 8.7e9  # GFLOPs


def fps_yolo_nano(gpu="A100"):
    """FPS estimados en GPU. A100 -> 200-300 FPS para YOLOv8-N."""
    if gpu == "A100":
        return 250
    elif gpu == "RTX3090":
        return 150
    elif gpu == "Jetson-Orin":
        return 60
    elif gpu == "CPU-i7":
        return 8
    return 0


def modelo_size_mb(params, bits_por_param=32):
    """Tamano del modelo en MB."""
    return params * bits_por_param / 8 / 1024 / 1024


def main() -> int:
    w = np.random.default_rng(0).normal(0, 1, size=(100, 100))
    q, scale, _ = main_cuantizar_int8(w)
    print(f"Original float32 size: {w.nbytes / 1024:.2f} KB")
    print(f"Cuantizado int8 size: {q.nbytes / 1024:.2f} KB")
    print(f"Scale: {scale:.4f}")
    # YOLO
    fps = fps_yolo_nano("Jetson-Orin")
    print(f"YOLOv8-N en Jetson-Orin: {fps} FPS")
    size = modelo_size_mb(3.2e6)  # 3.2M params
    print(f"Modelo 3.2M params: {size:.2f} MB")
    return 0


def main_cuantizar_int8(w):
    return cuantizar_int8(w)


if __name__ == "__main__":
    sys.exit(main())