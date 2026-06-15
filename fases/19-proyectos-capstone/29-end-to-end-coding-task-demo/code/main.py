"""
Lección: 29-end-to-end-coding-task-demo
Fase: 19
Capstone de ingeniería AI: 29 End To End Coding Task Demo.
"""
from __future__ import annotations
import sys

class CodingAgent:
    def __init__(self, llm, tools, sandbox):
        self.llm = llm
        self.tools = tools
        self.sandbox = sandbox

    def solve(self, issue):
        context = self.tools.explore(issue.repo)
        plan = self.llm.plan(issue, context)
        changes = self.tools.edit(plan)
        result = self.sandbox.test(changes)
        while not result["passes"] and result["iter"] < 5:
            changes = self.llm.fix(result, changes)
            result = self.sandbox.test(changes)
        return changes



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
