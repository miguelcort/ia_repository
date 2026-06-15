"""
Lección: 68-rag-eval-precision-recall
Fase: 19
Capstone de ingeniería AI: 68 Rag Eval Precision Recall.
"""
from __future__ import annotations
import sys

def context_precision(retrieved, relevant):
    return len(set(retrieved) & set(relevant)) / max(len(retrieved), 1)


def faithfulness(answer, context, judge_llm):
    return judge_llm(f"Context: {context}\nAnswer: {answer}\n"
                   f"Is the answer fully supported?")



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
