"""
Lección: 60-projection-layer-modality-align
Fase: 19
Capstone de ingeniería AI: 60 Projection Layer Modality Align.
"""
from __future__ import annotations
import sys

import torch.nn as nn


class VisionProjector(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(vision_dim, llm_dim),
                                 nn.GELU(),
                                 nn.Linear(llm_dim, llm_dim))

    def forward(self, vision_features):
        return self.proj(vision_features)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
