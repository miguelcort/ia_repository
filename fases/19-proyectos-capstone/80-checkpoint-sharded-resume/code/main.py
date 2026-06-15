"""
Lección: 80-checkpoint-sharded-resume
Fase: 19
Capstone de ingeniería AI: 80 Checkpoint Sharded Resume.
"""
from __future__ import annotations
import sys

from torch.distributed.checkpoint import save, load


def save_sharded(model, path):
    state = {"model": model.state_dict()}
    save(state, path)


def load_sharded(model, path):
    state = {"model": model.state_dict()}
    load(state, path)
    model.load_state_dict(state["model"])



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
