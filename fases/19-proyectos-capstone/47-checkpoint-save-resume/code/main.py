"""
Lección: 47-checkpoint-save-resume
Fase: 19
Capstone de ingeniería AI: 47 Checkpoint Save Resume.
"""
from __future__ import annotations
import sys

import torch
import os


def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    state = {"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "step": step, "loss": loss}
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(model, optimizer, scheduler, path):
    state = torch.load(path)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state["step"]



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
