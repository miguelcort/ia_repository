"""
Lección: 15-constitutional-safety-harness
Fase: 19
Capstone de ingeniería AI: 15 Constitutional Safety Harness.
"""
from __future__ import annotations
import sys

def safety_harness(prompt, llm, llamaguard, rules):
    """Safety pipeline."""
    response = llm.generate(prompt)
    if llamaguard.is_unsafe(response) or rules.violates(response):
        return {"status": "refused", "response": "I cannot help."}
    return {"status": "ok", "response": response}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
