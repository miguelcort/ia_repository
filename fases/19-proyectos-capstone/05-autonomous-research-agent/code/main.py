"""
Lección: 05-autonomous-research-agent
Fase: 19
Capstone de ingeniería AI: 05 Autonomous Research Agent.
"""
from __future__ import annotations
import sys

class ResearchAgent:
    def __init__(self, llm, retriever, runner):
        self.llm = llm
        self.retriever = retriever
        self.runner = runner
        self.hypotheses = []
        self.results = []

    def run(self, question, max_iters=5):
        for i in range(max_iters):
            hyp = self.llm.hypothesize(question)
            self.hypotheses.append(hyp)
            papers = self.retriever.search(hyp)
            result = self.runner.run(hyp)
            self.results.append(result)
            if self.is_satisfactory(result):
                return result
        return self.results[-1]



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
