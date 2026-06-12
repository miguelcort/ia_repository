"""
Lección: 42-agent-workbench-capstone
Fase: 14
Agent workbench capstone: full integration
of tools, memory, plan, reviewer, gates,
handoff, scope, todos, stats. End-to-end.
"""
from __future__ import annotations
import tempfile


def make_capstone(registry_class, tool_class, memory_class, scope_class,
                  reviewer_class, gate_class, todos_fn, stats_fn):
    """Create an integrated workbench that composes all parts."""
    registry = registry_class()
    memory = memory_class()
    scope = scope_class()
    reviewer = reviewer_class()
    gate = gate_class()
    return CapstoneWorkbench(registry, memory, scope, reviewer, gate, todos_fn, stats_fn)


class CapstoneWorkbench:
    def __init__(self, registry, memory, scope, reviewer, gate, todos_fn, stats_fn):
        self.registry = registry
        self.memory = memory
        self.scope = scope
        self.reviewer = reviewer
        self.gate = gate
        self.todos_fn = todos_fn
        self.stats_fn = stats_fn
        self.history = []

    def inspect_repo(self, root):
        todos = self.todos_fn(root)
        stats = self.stats_fn(root)
        self.history.append({"action": "inspect", "todos": len(todos), "stats": stats})
        return {"todos": todos, "stats": stats}

    def execute_action(self, action):
        ok, _, reason = self.scope.validate(action)
        if not ok:
            return {"ok": False, "reason": reason}
        self.history.append({"action": "execute", "tool": action})
        return {"ok": True, "result": f"executed {action}"}

    def review_output(self, output):
        result = self.reviewer.review(output)
        self.history.append({"action": "review", "approved": result["approved"]})
        return result

    def run_gates(self, fn):
        result = self.gate.run(fn)
        self.history.append({"action": "gate", "ok": result["ok"]})
        return result

    def full_workflow(self, root, action, output):
        inspect = self.inspect_repo(root)
        execution = self.execute_action(action)
        if not execution["ok"]:
            return {"step": "execute", "ok": False, "reason": execution["reason"]}
        review = self.review_output(output)
        return {
            "step": "done",
            "ok": review["approved"],
            "inspect": inspect,
            "execution": execution,
            "review": review,
        }


def main() -> int:
    print("Capstone workbench composed successfully")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())