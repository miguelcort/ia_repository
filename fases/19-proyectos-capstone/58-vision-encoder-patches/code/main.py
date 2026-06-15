"""
Lección: 58-vision-encoder-patches
Fase: 19
Capstone de ingeniería AI: 58 Vision Encoder Patches.
"""
from __future__ import annotations
import sys

import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, d=768):
        super().__init__()
        self.proj = nn.Conv2d(3, d, kernel_size=patch_size,
                             stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, num_patches, d))

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
