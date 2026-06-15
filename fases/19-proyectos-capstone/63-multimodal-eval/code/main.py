"""
Lección: 63-multimodal-eval
Fase: 19
Capstone de ingeniería AI: 63 Multimodal Eval.
"""
from __future__ import annotations
import sys

def vqa_eval(model, dataset):
    correct = 0
    for item in dataset:
        pred = model(image=item["image"],
                    question=item["question"])
        if pred == item["answer"]:
            correct += 1
    return correct / len(dataset)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
