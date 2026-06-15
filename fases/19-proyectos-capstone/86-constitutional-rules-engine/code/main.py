"""
Lección: 86-constitutional-rules-engine
Fase: 19
Capstone de ingeniería AI: 86 Constitutional Rules Engine.
"""
from __future__ import annotations
import sys

def check_compliance(response, constitution, judge_llm):
    for rule in constitution:
        score = judge_llm(f"Rule: {rule}\nResponse: {response}\n"
                         f"Does it comply?")
        if not score:
            return False, rule
    return True, None



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
