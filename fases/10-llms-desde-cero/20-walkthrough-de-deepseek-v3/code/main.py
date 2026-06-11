"""
Lección: 20-walkthrough-de-deepseek-v3
Fase: 10
DeepSeek-V3 (Feb 2025): 671B MoE, NSA, MLA, DualPipe, FP8, MTP.
SOTA open source, 14.8T tokens.
"""
from __future__ import annotations
import sys
import numpy as np


def deepseek_v3_architecture():
    """DeepSeek-V3 components."""
    return {
        "Total params": "671B",
        "Active per token": "37B (MoE, 8/256 experts)",
        "Attention": "MLA (Multi-Latent Attention, compressed KV)",
        "Sparse attention": "NSA (compression + selection + sliding)",
        "MoE": "256 experts, 8 active per token, shared experts",
        "Aux loss": "Load balancing (DeepSeek variant)",
        "FFN": "SwiGLU, experts",
        "Norm": "RMSNorm (pre-norm)",
        "RoPE": "Standard, YaRN extension",
        "Tokenizer": "BBPE 128K, multilingual",
        "Context": "128K",
        "MTP": "Multi-Token Prediction training signal",
        "Quant": "FP8 training, INT4 inference (AWQ)",
        "Parallelism": "DualPipe + TP + PP + DP + EP (expert parallel)",
        "Data": "14.8T tokens, multi-source",
    }


def deepseek_v3_costs():
    """Costos y benchmark."""
    return {
        "Training tokens": "14.8T",
        "GPU hours": "2.788M H800 hours",
        "Total cost": "~$5-10M (open source, 1/10 de closed frontier)",
        "Performance": "Comparable o > Llama 3.1 405B Instruct",
        "Open source": "MIT license (v3.1)",
        "v3 variants": "Base, Instruct, R1 (reasoning)",
    }


def moe_routing(top_k=8, n_experts=256, n_shared=2):
    """MoE routing: top_k experts + shared experts.
    Total active = top_k + n_shared.
    """
    return top_k + n_shared


def main() -> int:
    print("=== DeepSeek-V3 architecture ===")
    for k, v in deepseek_v3_architecture().items():
        print(f"  {k:18s} {v}")
    active = moe_routing(top_k=8, n_experts=256, n_shared=2)
    print(f"\nMoE active: {active}/256 experts per token (37B/671B)")
    return 0


if __name__ == "__main__":
    sys.exit(main())