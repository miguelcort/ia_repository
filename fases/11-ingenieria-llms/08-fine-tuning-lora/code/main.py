"""
Lección: 08-fine-tuning-lora
Fase: 11
Fine-tuning LLMs: LoRA, QLoRA, full FT. PEFT, TRL, axolotl.
"""
from __future__ import annotations
import sys
import numpy as np


def lora_init(in_dim, out_dim, rank=4, alpha=1.0, seed=0):
    """LoRA: A in (in, r), B in (r, out). A random, B=0."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((in_dim, rank))
    B = np.zeros((rank, out_dim))
    return A, B


def lora_forward(x, W, A, B, alpha=1.0, rank=4):
    """y = x @ W + (alpha/rank) * x @ A @ B."""
    base = x @ W
    lora_update = (alpha / rank) * (x @ A @ B)
    return base + lora_update


def lora_merge(W, A, B, alpha=1.0, rank=4):
    """Merge LoRA en W: W_new = W + (alpha/rank) * A @ B."""
    return W + (alpha / rank) * (A @ B)


def lora_trainable_params(in_dim, out_dim, rank):
    """Numero de params entrenables en LoRA."""
    return in_dim * rank + rank * out_dim


def full_ft_params(in_dim, out_dim):
    """Params full FT."""
    return in_dim * out_dim


def qlora_config(bits=4, double_quant=True, nf4=True):
    """QLoRA config."""
    return {
        "load_in_4bit": bits == 4,
        "load_in_8bit": bits == 8,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4" if nf4 else "fp4",
        "bnb_4bit_use_double_quant": double_quant,
    }


def main() -> int:
    in_dim, out_dim, rank = 64, 128, 8
    lora_params = lora_trainable_params(in_dim, out_dim, rank)
    full_params = full_ft_params(in_dim, out_dim)
    print(f"LoRA params: {lora_params}")
    print(f"Full params: {full_params}")
    print(f"Reduction: {full_params / lora_params:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())