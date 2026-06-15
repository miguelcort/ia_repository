"""
Lección: 76-collective-ops-from-scratch
Fase: 19
Capstone de ingeniería AI: 76 Collective Ops From Scratch.
"""
from __future__ import annotations
import sys

import torch.distributed as dist


def all_reduce(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()


def all_gather(tensor):
    world = dist.get_world_size()
    out = [torch.zeros_like(tensor) for _ in range(world)]
    dist.all_gather(out, tensor)
    return out



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
