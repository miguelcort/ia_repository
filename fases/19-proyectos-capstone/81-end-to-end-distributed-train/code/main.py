"""
Lección: 81-end-to-end-distributed-train
Fase: 19
Capstone de ingeniería AI: 81 End To End Distributed Train.
"""
from __future__ import annotations
import sys

def main():
    import torch.distributed as dist
    dist.init_process_group("nccl")
    model = setup_fsdp_zero3(Llama3_70B())
    loader = setup_dataloader()
    for step, batch in enumerate(loader):
        loss = train_step(model, batch)
        if step % 1000 == 0:
            save_sharded(model, f"ckpt-{step}/")



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
