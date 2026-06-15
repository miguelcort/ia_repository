"""
Lección: 24-plan-execute-control-flow
Fase: 19
Capstone de ingeniería AI: 24 Plan Execute Control Flow.
"""
from __future__ import annotations
import sys

class PlanExecuteAgent:
    def run(self, goal, max_replans=3):
        for _ in range(max_replans):
            plan = self.llm.plan(goal, self.context)
            results = []
            for step in plan:
                result = self.tools.execute(step)
                results.append(result)
                if self.is_failed(result):
                    break
            if self.all_succeeded(results):
                return self.synthesize(results)
            self.context.update(plan, results)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
