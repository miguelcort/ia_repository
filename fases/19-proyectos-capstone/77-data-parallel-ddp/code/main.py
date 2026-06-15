"""
Lección: 77-data-parallel-ddp
Fase: 19
Capstone de ingeniería AI: 77 Data Parallel Ddp.
"""
from __future__ import annotations
import sys

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def ddp_train(rank, world_size, model_fn):
    dist.init_process_group("nccl", rank=rank,
                            world_size=world_size)
    torch.cuda.set_device(rank)
    model = model_fn().to(rank)
    return DDP(model, device_ids=[rank])



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
