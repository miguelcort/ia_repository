"""
Lección: 11-handoffs-and-routines
Fase: 16
Handoffs and routines: transfer of
control between agents, multi-step
routines, OpenAI Agents SDK pattern,
routing logic, shared context.
"""
from __future__ import annotations
import uuid


class Handoff:
    def __init__(self, from_agent, to_agent, reason, context=None):
        self.id = str(uuid.uuid4())
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.reason = reason
        self.context = context or {}


def handoff(agent_id, target, reason="", context=None):
    return Handoff(agent_id, target, reason, context)


class Routine:
    def __init__(self, name, steps):
        self.name = name
        self.steps = list(steps)
        self.cursor = 0
        self.history = []

    def current_step(self):
        if self.cursor >= len(self.steps):
            return None
        return self.steps[self.cursor]

    def advance(self, result=None):
        step = self.current_step()
        if step is None:
            return None
        self.history.append({"step": step, "result": result})
        self.cursor += 1
        return self.current_step()

    def is_done(self):
        return self.cursor >= len(self.steps)

    def run(self, executor):
        """executor(step) -> result."""
        while not self.is_done():
            step = self.current_step()
            result = executor(step)
            self.advance(result)
        return self.history


def handoff_to_routine(agent_id, target, routine_name, reason=""):
    return {
        "type": "handoff_with_routine",
        "from": agent_id,
        "to": target,
        "routine": routine_name,
        "reason": reason,
    }


def main() -> int:
    h = handoff("a1", "a2", "specialist needed", {"topic": "x"})
    print(f"Handoff: {h.from_agent} -> {h.to_agent}")
    r = Routine("onboard", ["greet", "ask_name", "ask_email"])
    print(f"Steps: {len(r.steps)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())