"""
Lección: 35-gpt-model-assembly
Fase: 19
Capstone de ingeniería AI: 35 Gpt Model Assembly.
"""
from __future__ import annotations
import sys

import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, vocab, d, n_heads, n_layers, max_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, n_heads, 4 * d)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, ids):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
