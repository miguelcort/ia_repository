"""
Lección: 41-eval-pipeline
Fase: 19
Capstone de ingeniería AI: 41 Eval Pipeline.
"""
from __future__ import annotations
import sys

import lm_eval
from lm_eval.models.huggingface import HFLM


def run_lm_eval(model_id, tasks, batch_size=4):
    model = HFLM(pretrained=model_id, batch_size=batch_size)
    return lm_eval.simple_evaluate(model=model, tasks=tasks,
                                   batch_size=batch_size)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
