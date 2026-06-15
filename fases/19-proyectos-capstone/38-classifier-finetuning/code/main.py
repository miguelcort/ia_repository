"""
Lección: 38-classifier-finetuning
Fase: 19
Capstone de ingeniería AI: 38 Classifier Finetuning.
"""
from __future__ import annotations
import sys

import torch.nn as nn
from transformers import AutoModel


class Classifier(nn.Module):
    def __init__(self, model_id, n_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        d = self.encoder.config.hidden_size
        self.head = nn.Linear(d, n_classes)

    def forward(self, ids, mask):
        out = self.encoder(ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0]
        return self.head(cls)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
