"""
Lección: 05-context-engineering
Fase: 11
Context engineering: optimizar context window, dynamic context, memory.
vs prompt engineering.
"""
from __future__ import annotations
import sys
import numpy as np


def count_tokens_simple(text, chars_per_token=4):
    return len(text) // chars_per_token


def context_budget_check(prompt, max_tokens=8192):
    """Check si prompt excede el context budget."""
    n = count_tokens_simple(prompt)
    return n <= max_tokens, n


def context_priority(items, max_tokens):
    """Context prioritization: items con priority + content.
    Greedy fill hasta max_tokens.
    """
    sorted_items = sorted(items, key=lambda x: -x.get("priority", 0))
    used = 0
    out = []
    for item in sorted_items:
        cost = count_tokens_simple(item["content"])
        if used + cost <= max_tokens:
            out.append(item)
            used += cost
    return out, used


def sliding_context(context, max_tokens):
    """Sliding window: mantener ultimos max_tokens tokens.
    Asume tokens separados por espacio.
    """
    tokens = context.split()
    if len(tokens) <= max_tokens:
        return context
    return " ".join(tokens[-max_tokens:])


def main() -> int:
    ok, n = context_budget_check("Hola " * 100, max_tokens=200)
    print(f"Budget ok: {ok}, tokens: {n}")
    items = [
        {"content": "system: " + "x" * 200, "priority": 10},
        {"content": "context: " + "x" * 500, "priority": 5},
        {"content": "history: " + "x" * 1000, "priority": 1},
    ]
    selected, used = context_priority(items, max_tokens=200)
    print(f"Selected: {len(selected)} items, used: {used} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())