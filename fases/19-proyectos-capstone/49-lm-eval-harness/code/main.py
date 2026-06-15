"""
Lección: 49-lm-eval-harness
Fase: 19
Capstone de ingeniería AI: 49 Lm Eval Harness.
"""
from __future__ import annotations
import sys

import lm_eval
from lm_eval.models.huggingface import HFLM


def run_mmlu(model_id, n_shot=5):
    model = HFLM(pretrained=model_id)
    results = lm_eval.simple_evaluate(model=model, tasks=["mmlu"],
                                      num_fewshot=n_shot)
    return results["results"]["mmlu"]



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
