"""
Lección: 85-content-classifier-integration
Fase: 19
Capstone de ingeniería AI: 85 Content Classifier Integration.
"""
from __future__ import annotations
import sys

def moderation_pipeline(prompt, llm, llamaguard, openai_mod):
    if llamaguard.is_unsafe(prompt) or openai_mod.flagged(prompt):
        return {"blocked": True, "stage": "input"}
    response = llm(prompt)
    if llamaguard.is_unsafe(response) or openai_mod.flagged(response):
        return {"blocked": True, "stage": "output"}
    return {"blocked": False, "response": response}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
