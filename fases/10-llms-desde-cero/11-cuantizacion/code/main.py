"""
Lección: 11-cuantizacion
Fase: 10
Quantization: post-training (PTQ) y quantization-aware training (QAT).
INT8, INT4, FP8, NF4. GPTQ, AWQ, SmoothQuant, bitsandbytes.
"""
from __future__ import annotations
import sys
import numpy as np


def absmax_quantize(x, num_bits=8):
    """Abamax quantization: x_int = round(x / abs_max * (2^(n-1) - 1)).
    Symmetric, per-tensor.
    """
    abs_max = np.abs(x).max()
    if abs_max == 0:
        return x.astype(np.int8), 1.0
    scale = (2 ** (num_bits - 1) - 1) / abs_max
    x_q = np.round(x * scale).clip(-(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1)
    return x_q.astype(np.int8), 1.0 / scale


def dequantize(x_q, scale):
    """Dequantize: x = x_q * scale."""
    return x_q.astype(np.float32) * scale


def nf4_quantize(x, num_bits=4):
    """NF4 (4-bit NormalFloat, QLoRA). Simulate quantile-based.
    Para demo, usamos 16 levels uniform.
    """
    abs_max = np.abs(x).max()
    if abs_max == 0:
        return x.astype(np.int8), 1.0
    # 16 levels in [-1, 1]
    levels = np.linspace(-1, 1, 16)
    normalized = x / abs_max
    # Quantize to nearest level
    x_q = np.digitize(normalized, levels) - 1
    return x_q.astype(np.int8), abs_max


def gptq_per_row(x, W, block_size=128):
    """GPTQ-style: per-row quantization of weight matrix.
    Simplificado: absmax per row.
    """
    out = np.zeros_like(W, dtype=np.int8)
    scales = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        row = W[i]
        abs_max = np.abs(row).max()
        if abs_max == 0:
            scales[i] = 1.0
            continue
        scale = 127.0 / abs_max
        scales[i] = 1.0 / scale
        out[i] = np.round(row * scale).clip(-128, 127).astype(np.int8)
    return out, scales


def awq_per_channel(x, num_bits=4):
    """AWQ: activation-aware weight quantization.
    Simplificado: per-channel absmax.
    """
    out = np.zeros_like(x, dtype=np.int8)
    scales = np.zeros(x.shape[0]) if x.ndim > 1 else np.array([1.0])
    for i in range(x.shape[0]):
        row = x[i] if x.ndim > 1 else x
        abs_max = np.abs(row).max()
        if abs_max == 0:
            continue
        scale = 7.0 / abs_max
        scales[i] = 1.0 / scale
        out[i] = np.round(row * scale).clip(-8, 7).astype(np.int8)
    return out, scales


def memory_savings(orig_bytes, num_bits=4):
    """Memory savings: orig_bytes / (num_bits/8)."""
    return orig_bytes * num_bits // 8


def quant_methods():
    return {
        "PTQ (Post-Training Quantization)": "Quantize despues de training. GPTQ, AWQ",
        "QAT (Quantization-Aware Training)": "Train con quantization simulation. +quality",
        "INT8": "8-bit weights/activations. 2x memory reduction",
        "INT4": "4-bit weights. 4x reduction. Quality loss",
        "FP8": "8-bit float. E4M3, E5M2. H100 native",
        "NF4": "NormalFloat 4-bit. QLoRA. +quality vs INT4",
    }


def main() -> int:
    # Demo
    x = np.random.default_rng(0).standard_normal((4, 4)) * 2
    x_q, scale = absmax_quantize(x, num_bits=8)
    print(f"Original: dtype={x.dtype}, mem={x.nbytes}")
    print(f"Quantized: dtype={x_q.dtype}, mem={x_q.nbytes}, scale={scale:.3f}")
    # Dequantize
    x_dq = dequantize(x_q, scale)
    err = np.abs(x - x_dq).max()
    print(f"Max abs error: {err:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())