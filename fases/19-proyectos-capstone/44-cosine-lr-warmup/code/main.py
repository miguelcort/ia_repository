"""
Lección: 44-cosine-lr-warmup
Fase: 19
Capstone de ingeniería AI: 44 Cosine Lr Warmup.
"""
from __future__ import annotations
import sys

import math


def lr_schedule(step, warmup, max_steps, max_lr, min_lr=0):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (
        1 + math.cos(math.pi * progress))



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
