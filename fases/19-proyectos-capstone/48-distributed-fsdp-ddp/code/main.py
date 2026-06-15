"""
Lección: 48-distributed-fsdp-ddp
Fase: 19
Capstone de ingeniería AI: 48 Distributed Fsdp Ddp.
"""
from __future__ import annotations
import sys

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision


def fsdp_setup(model, use_mp=True):
    mp_policy = (MixedPrecision(param_dtype=torch.bfloat16,
                                reduce_dtype=torch.bfloat16,
                                buffer_dtype=torch.bfloat16)
                if use_mp else None)
    return FSDP(model, mixed_precision=mp_policy, use_orig_params=True)


def ddp_setup(rank, world_size, backend="nccl"):
    dist.init_process_group(backend, rank=rank,
                            world_size=world_size)
    torch.cuda.set_device(rank)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
