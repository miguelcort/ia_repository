"""
Lección: 36-training-loop-eval
Fase: 19
Capstone de ingeniería AI: 36 Training Loop Eval.
"""
from __future__ import annotations
import sys

import torch.nn.functional as F


def train_step(model, batch, optimizer, grad_clip=1.0):
    ids, targets = batch
    logits = model(ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                          targets.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


@torch.no_grad()
def eval_pass(model, val_loader):
    total_loss = 0
    n = 0
    for ids, targets in val_loader:
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              targets.view(-1))
        total_loss += loss.item() * ids.size(0)
        n += ids.size(0)
    return {"perplexity": (total_loss / n) ** 0.5}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
