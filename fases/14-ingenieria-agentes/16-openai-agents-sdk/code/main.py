"""
Lección: 16-openai-agents-sdk
Fase: 14
OpenAI Agents SDK (2024): handoffs, tools, tracing, guardrails.
Built on top of Chat Completions. Production agent framework.
+Tracing, +guardrails, +handoffs.
"""
from __future__ import annotations
from collections import defaultdict


class Agent:
    """Mock OpenAI Agent."""
    def __init__(self, name, instructions, tools=None, handoffs=None, model="gpt-4o"):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        self.handoffs = handoffs or []  # other agents to hand off to
        self.model = model
        self.calls = []
        self.handoff_count = 0

    def run(self, input_text, session=None):
        """Run agent on input."""
        self.calls.append({"input": input_text, "session": session})
        # decide: handoff or respond
        if self.handoffs and "transfer" in input_text.lower():
            target = self.handoffs[0]
            self.handoff_count += 1
            return {"handoff_to": target.name, "agent": self.name}
        return {"response": f"[{self.name}] processed: {input_text[:50]}", "agent": self.name}

    def add_handoff(self, agent):
        self.handoffs.append(agent)


class Runner:
    """Mock OpenAI Runner: orchestrates agent loop."""
    def __init__(self, agent, tracing=True):
        self.agent = agent
        self.tracing = tracing
        self.traces = []

    def run(self, input_text, max_turns=10):
        """Run agent loop with handoffs."""
        current = self.agent
        history = []
        for turn in range(max_turns):
            if self.tracing:
                self.traces.append({"agent": current.name, "input": input_text})
            result = current.run(input_text)
            history.append(result)
            if "handoff_to" in result:
                # find target
                next_agent = next((a for a in current.handoffs if a.name == result["handoff_to"]), None)
                if next_agent is None:
                    return history
                current = next_agent
                input_text = result.get("response", "")
            else:
                break
        return history


class Guardrail:
    """Mock input/output guardrail."""
    def __init__(self, name, check_fn):
        self.name = name
        self.check_fn = check_fn

    def validate(self, text):
        return self.check_fn(text)


def main() -> int:
    triage = Agent("Triage", "Triage user requests")
    billing = Agent("Billing", "Handle billing issues")
    triage.add_handoff(billing)
    runner = Runner(triage, tracing=True)
    results = runner.run("I need to transfer to billing")
    print(f"Results: {len(results)}, traces: {len(runner.traces)}")
    print(f"Final: {results[-1]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())