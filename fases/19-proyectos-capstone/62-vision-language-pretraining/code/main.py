"""
Lección: 62-vision-language-pretraining
Fase: 19
Capstone de ingeniería AI: 62 Vision Language Pretraining.
"""
from __future__ import annotations
import sys

import torch
import torch.nn.functional as F


def clip_loss(image_embeds, text_embeds, temperature=0.07):
    logits = image_embeds @ text_embeds.T / temperature
    labels = torch.arange(logits.size(0))
    return (F.cross_entropy(logits, labels)
           + F.cross_entropy(logits.T, labels)) / 2


def siglip_loss(image_embeds, text_embeds, temperature=10.0):
    logits = image_embeds @ text_embeds.T * temperature
    labels = torch.eye(logits.size(0)) * 2 - 1
    return -F.logsigmoid(labels * logits).sum() / logits.size(0)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
