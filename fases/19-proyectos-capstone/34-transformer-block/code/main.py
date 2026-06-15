"""
Lección: 34-transformer-block
Fase: 19
Capstone de ingeniería AI: 34 Transformer Block.
"""
from __future__ import annotations
import sys

import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads,
                                          batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x),
                         self.ln1(x), need_weights=False)[0]
        return x + self.mlp(self.ln2(x))



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
