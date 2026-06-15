"""
Lección: 20-agent-harness-loop-contract
Fase: 19
Capstone de ingeniería AI: 20 Agent Harness Loop Contract.
"""
from __future__ import annotations
import sys

class AgentHarness:
    def __init__(self, llm, tools, budget):
        self.llm = llm
        self.tools = tools
        self.budget = budget
        self.state = {"todos": [], "history": []}

    def run(self, prompt, max_turns=20):
        for turn in range(max_turns):
            if self.budget.exceeded():
                break
            self.state = self.llm.update_plan(prompt, self.state)
            tool_call = self.llm.decide_tool(self.state)
            if tool_call is None:
                return self.state["final_answer"]
            result = self.tools.execute(tool_call)
            self.state["history"].append((tool_call, result))
        return self.state.get("final_answer", "")



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
