"""
Lección: 50-hypothesis-generator
Fase: 19
Capstone de ingeniería AI: 50 Hypothesis Generator.
"""
from __future__ import annotations
import sys

def generate_hypothesis(topic, prior_lit, llm):
    prompt = (f"Topic: {topic}\nPrior: {prior_lit}\n"
             f"Generate 3 testable hypotheses with "
             f"novelty, feasibility, impact.")
    return llm.generate(prompt)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
