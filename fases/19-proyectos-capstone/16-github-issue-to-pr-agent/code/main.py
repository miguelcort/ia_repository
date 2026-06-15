"""
Lección: 16-github-issue-to-pr-agent
Fase: 19
Capstone de ingeniería AI: 16 Github Issue To Pr Agent.
"""
from __future__ import annotations
import sys

class IssueToPRAgent:
    def run(self, issue):
        plan = self.llm.plan(issue, self.repo_context)
        branch = self.create_branch(plan.target_files)
        changes = self.apply_changes(plan)
        tests = self.generate_tests(changes)
        result = self.run_tests(tests)
        return self.open_pr(branch, changes, plan, result)



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
