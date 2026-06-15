"""
Lección: 40-dpo-from-scratch
Fase: 19
Capstone de ingeniería AI: 40 Dpo From Scratch.
"""
from __future__ import annotations
import sys

import torch.nn.functional as F


def dpo_loss(policy_chosen, policy_rejected, ref_chosen,
           ref_rejected, beta=0.1):
    diff = beta * ((policy_chosen - ref_chosen)
                  - (policy_rejected - ref_rejected))
    return -F.logsigmoid(diff).mean()



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
