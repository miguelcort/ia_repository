"""
Lección: 27-eval-harness-fixture-tasks
Fase: 19
Capstone de ingeniería AI: 27 Eval Harness Fixture Tasks.
"""
from __future__ import annotations
import sys

def run_eval(agent, tasks):
    results = []
    for task in tasks:
        output = agent.run(task["input"])
        score = task["scoring"](output, task["expected"])
        results.append({"task_id": task["id"], "score": score,
                       "output": output})
    return {"mean_score": sum(r["score"] for r in results) / max(len(results), 1),
            "results": results}



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
