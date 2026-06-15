"""
Lección: 17-personal-ai-tutor
Fase: 19
Capstone de ingeniería AI: 17 Personal Ai Tutor.
"""
from __future__ import annotations
import sys

class PersonalTutor:
    def __init__(self, llm, kt_model):
        self.llm = llm
        self.kt = kt_model
        self.profile = {}

    def teach(self, topic, question):
        mastery = self.kt.estimate(topic, self.profile)
        if mastery < 0.7:
            return self.socratic(topic, question)
        return self.advance(topic, question)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
