"""
Lección: 57-end-to-end-research-demo
Fase: 19
Capstone de ingeniería AI: 57 End To End Research Demo.
"""
from __future__ import annotations
import sys

class ResearchPipeline:
    def __init__(self, llm, sandbox, evaluator):
        self.llm = llm
        self.sandbox = sandbox
        self.evaluator = evaluator

    def run(self, question, max_iters=5):
        hypotheses = self.generate_hypotheses(question)
        for i in range(max_iters):
            results = [self.sandbox.run(self.generate_code(h))
                      for h in hypotheses]
            best = self.evaluator.rank(results)
            hypotheses = self.refine(hypotheses, best)
        return self.write_paper(hypotheses, results)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
