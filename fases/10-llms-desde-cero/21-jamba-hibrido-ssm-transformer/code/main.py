"""
Lección: 21-jamba-hibrido-ssm-transformer
Fase: 10
Jamba (AI21 2024): hybrid Mamba + transformer. 12B / 52B MoE, 256K context.
Combina attention + SSM para long context efficiency.
"""
from __future__ import annotations
import sys
import numpy as np


def jamba_architecture():
    """Jamba architecture."""
    return {
        "Params": "12B (Jamba 1.0), 52B (Jamba 1.5 Large, MoE)",
        "Hybrid": "8 transformer blocks + 7 mamba blocks (alternating)",
        "MoE (1.5)": "MoE layers en Jamba 1.5",
        "Context": "256K tokens",
        "Attention": "GQA, sliding window + global",
        "Mamba": "State-space model, linear O(n)",
        "Normalization": "RMSNorm (pre-norm)",
        "Tokenizer": "BBPE 64K (Jamba 1.0), 100K (1.5)",
        "Quantization": "INT4 AWQ",
    }


def hybrid_architecture_ratios():
    """Jamba ratios: transformer vs mamba."""
    return {
        "Jamba 1.0 mini": "4 transformer + 28 mamba",
        "Jamba 1.0": "8 transformer + 28 mamba",
        "Jamba 1.5 large": "Mix, MoE + transformer + mamba",
    }


def ssm_complexity(seq_len, dim, state_size=16):
    """Mamba SSM: O(n * dim * state_size) compute, O(dim * state_size) memory."""
    compute = seq_len * dim * state_size
    memory = dim * state_size
    return compute, memory


def attention_complexity(seq_len):
    """Standard attention: O(n^2) memory and compute."""
    return seq_len ** 2


def main() -> int:
    print("=== Jamba architecture ===")
    for k, v in jamba_architecture().items():
        print(f"  {k:18s} {v}")
    # Complexity comparison
    n = 128000
    ssm_c, ssm_m = ssm_complexity(n, dim=4096)
    attn_c = attention_complexity(n)
    print(f"\n=== Complexity at 128K context ===")
    print(f"  SSM (Mamba): compute={ssm_c:.2e}, memory={ssm_m:.2e}")
    print(f"  Attention: compute/memory={attn_c:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())