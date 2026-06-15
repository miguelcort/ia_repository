"""
Lección: 59-vit-transformer
Fase: 19
Capstone de ingeniería AI: 59 Vit Transformer.
"""
from __future__ import annotations
import sys

import torch
import torch.nn as nn


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, d=768,
                n_heads=12, n_layers=12, n_classes=1000):
        super().__init__()
        self.embed = PatchEmbed(img_size, patch_size, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, n_heads, 4 * d)
             for _ in range(n_layers)])
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        x = self.embed(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
