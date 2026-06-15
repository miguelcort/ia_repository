"""
Lección: 33-multihead-self-attention
Fase: 19
Capstone de ingeniería AI: 33 Multihead Self Attention.
"""
from __future__ import annotations
import sys

import torch
import torch.nn.functional as F


def multihead_attention(x, W_qkv, W_o, n_heads, causal=True):
    B, T, D = x.shape
    qkv = x @ W_qkv
    q, k, v = qkv.chunk(3, dim=-1)
    H = n_heads
    Dh = D // H
    q = q.view(B, T, H, Dh).transpose(1, 2)
    k = k.view(B, T, H, Dh).transpose(1, 2)
    v = v.view(B, T, H, Dh).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v,
                                          is_causal=causal)
    return out.transpose(1, 2).reshape(B, T, D) @ W_o



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
