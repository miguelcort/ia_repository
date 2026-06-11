"""
Lección: 02-few-shot-y-cot
Fase: 11
Few-shot learning, Chain-of-Thought, self-consistency.
"""
from __future__ import annotations
import sys
import numpy as np


def few_shot_prompt(instruction, examples, format_str="Q: {q}\nA: {a}"):
    """Build few-shot prompt."""
    parts = []
    for ex in examples:
        parts.append(format_str.format(q=ex[0], a=ex[1]))
    parts.append(format_str.format(q=instruction, a=""))
    return "\n\n".join(parts)


def cot_prompt(question):
    """Chain-of-thought: anade 'razona paso a paso'."""
    return f"Q: {question}\n\nA: Razona paso a paso antes de responder.\n\nPaso 1:"


def self_consistency_answer(answers):
    """Self-consistency: majority vote.
    answers: list of strings.
    Returns: most common.
    """
    from collections import Counter
    if not answers:
        return ""
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def diversity_score(answers):
    """Diversity: unique / total."""
    if not answers:
        return 0.0
    return len(set(answers)) / len(answers)


def zero_shot_cot(question):
    """Zero-shot CoT: solo magic phrase.
    Kojima 2022: 'Let's think step by step'.
    """
    return f"Q: {question}\n\nA: Pensemos paso a paso."


def main() -> int:
    examples = [("2+2", "4"), ("3+3", "6")]
    p = few_shot_prompt("5+5", examples)
    print(f"Few-shot:\n{p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())