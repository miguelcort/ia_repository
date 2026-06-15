"""
Lección: 45-gradient-clipping-amp
Fase: 19
Capstone de ingeniería AI: 45 Gradient Clipping Amp.
"""
from __future__ import annotations
import sys

import torch
from torch.cuda.amp import autocast


def train_step_amp(model, batch, optimizer):
    ids, targets = batch
    with autocast(dtype=torch.bfloat16):
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              targets.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
