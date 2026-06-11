"""
Lección: 01-prompt-engineering
Fase: 11
Prompt engineering: zero/few-shot, CoT, ReAct, system prompts.
"""
from __future__ import annotations
import sys
import numpy as np


def build_prompt(instruction, context=None, examples=None, system=None):
    """Build a prompt with optional components."""
    parts = []
    if system:
        parts.append(f"### System:\n{system}\n")
    if context:
        parts.append(f"### Context:\n{context}\n")
    if examples:
        for ex in examples:
            parts.append(f"### Example:\nInput: {ex[0]}\nOutput: {ex[1]}\n")
    parts.append(f"### Instruction:\n{instruction}\n\n### Response:\n")
    return "\n".join(parts)


def cot_prompt(question):
    """Chain-of-thought: anade 'razona paso a paso'."""
    return f"{question}\n\nRazona paso a paso antes de responder."


def react_prompt(question, tools):
    """ReAct: razonamiento + acciones.
    tools: list of (name, description).
    """
    tool_str = "\n".join(f"- {n}: {d}" for n, d in tools)
    return f"Question: {question}\n\nTools disponibles:\n{tool_str}\n\nUsa el formato:\nThought: <razonamiento>\nAction: <tool_name>\nObservation: <resultado>\n... (Thought/Action/Observation)\nFinal Answer: <respuesta>"


def prompt_tokens_estimate(prompt, chars_per_token=4):
    """Estimate token count from prompt length."""
    return len(prompt) // chars_per_token


def main() -> int:
    p = build_prompt("Cual es la capital de Francia?")
    print(f"Basic prompt:\n{p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())