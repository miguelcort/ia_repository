"""
Lección: 61-cross-attention-fusion
Fase: 19
Capstone de ingeniería AI: 61 Cross Attention Fusion.
"""
from __future__ import annotations
import sys

import torch
import torch.nn as nn


class QFormer(nn.Module):
    def __init__(self, n_queries=32, vision_dim=1024, d=768,
                n_layers=6, n_heads=12):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d))
        self.vision_proj = nn.Linear(vision_dim, d)
        self.layers = nn.ModuleList(
            [TransformerBlock(d, n_heads, 4 * d)
             for _ in range(n_layers)])

    def forward(self, vision_features):
        v = self.vision_proj(vision_features)
        q = self.queries.expand(v.size(0), -1, -1)
        for layer in self.layers:
            q = layer.cross_attn(q, v, v)
        return q



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
