"""
Lección: 14-walkthroughs-de-arquitectura-de-modelos-open
Fase: 10
Walkthroughs: Llama 3, Mistral, Qwen 2, Phi-3, Gemma 2, SmolLM.
Componentes: attention, FFN, normalization, embeddings.
"""
from __future__ import annotations
import sys
import numpy as np


def llama3_architecture():
    """Llama 3 (Meta 2024)."""
    return {
        "Params": "8B, 70B, 405B",
        "Attention": "GQA (8 K/V heads para 8B, 8 para 70B)",
        "RoPE": "Base 500K, theta",
        "FFN": "SwiGLU (3 matrices, d_ff ~ 2.7 * d)",
        "Norm": "RMSNorm (pre-norm)",
        "Tokenizer": "BBPE 128K, multilingual",
        "Context": "8K (128K con YaRN)",
        "Vocab": "128K tokens",
    }


def mistral_architecture():
    """Mistral 7B."""
    return {
        "Params": "7B",
        "Attention": "GQA-8 (4 K/V heads)",
        "RoPE": "theta=1e6",
        "FFN": "SwiGLU",
        "Norm": "RMSNorm",
        "Tokenizer": "BBPE 32K",
        "Sliding window": "Window 4096",
        "Context": "8K (32K con YaRN)",
    }


def qwen_architecture():
    """Qwen 2.5 (Alibaba 2024)."""
    return {
        "Params": "0.5B, 1.5B, 7B, 72B",
        "Attention": "GQA (4-8 K/V heads)",
        "RoPE": "theta=1M",
        "FFN": "SwiGLU",
        "Norm": "RMSNorm",
        "Tokenizer": "BBPE 152K (multilingual)",
        "Context": "128K",
    }


def phi_architecture():
    """Phi-3 (Microsoft 2024)."""
    return {
        "Params": "mini (3.8B), small (7B), medium (14B)",
        "Attention": "Multi-head (no GQA)",
        "RoPE": "theta=10K",
        "FFN": "SwiGLU",
        "Norm": "RMSNorm",
        "Tokenizer": "BBPE 32K",
        "Context": "4K-128K",
    }


def gemma_architecture():
    """Gemma 2 (Google 2024)."""
    return {
        "Params": "2B, 9B, 27B",
        "Attention": "GQA (4 K/V heads)",
        "RoPE": "Base 10K",
        "FFN": "GeGLU (SwiGLU variant)",
        "Norm": "RMSNorm + GeGLU",
        "Tokenizer": "SentencePiece 256K",
    }


def smollm_architecture():
    """SmolLM (HuggingFace 2024)."""
    return {
        "Params": "135M, 360M, 1.7B",
        "Attention": "GQA",
        "RoPE": "Standard",
        "FFN": "SwiGLU",
        "Norm": "RMSNorm",
        "Tokenizer": "BBPE 32K",
    }


def comparison_table():
    """Comparacion de modelos SOTA."""
    return [
        ["Llama 3 8B", "8B", "128K", "GQA-8", "8K (128K YaRN)"],
        ["Llama 3 70B", "70B", "128K", "GQA-8", "8K (128K YaRN)"],
        ["Mistral 7B", "7B", "32K", "GQA-4", "8K (32K YaRN)"],
        ["Qwen 2.5 72B", "72B", "152K", "GQA-8", "128K"],
        ["Phi-3 medium", "14B", "32K", "MHA", "4K-128K"],
        ["Gemma 2 27B", "27B", "256K", "GQA-4", "8K"],
        ["SmolLM 1.7B", "1.7B", "32K", "GQA", "2K-8K"],
    ]


def main() -> int:
    print("=== Model architecture comparison ===")
    for row in comparison_table():
        print(f"  {row[0]:20s} {row[1]:8s} vocab={row[2]:8s} {row[3]:10s} ctx={row[4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())