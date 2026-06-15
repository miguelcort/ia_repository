"""
Lección: 46-gradient-accumulation
Fase: 19
Capstone de ingeniería AI: 46 Gradient Accumulation.
"""
from __future__ import annotations
import sys

def train_with_accum(model, batches, optimizer, accumulation_steps=8):
    optimizer.zero_grad()
    for i, batch in enumerate(batches):
        ids, targets = batch
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              targets.view(-1))
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
