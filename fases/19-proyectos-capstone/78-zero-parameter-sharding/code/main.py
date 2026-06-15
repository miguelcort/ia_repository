"""
Lección: 78-zero-parameter-sharding
Fase: 19
Capstone de ingeniería AI: 78 Zero Parameter Sharding.
"""
from __future__ import annotations
import sys

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


def setup_fsdp_zero3(model):
    return FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
               use_orig_params=True)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
