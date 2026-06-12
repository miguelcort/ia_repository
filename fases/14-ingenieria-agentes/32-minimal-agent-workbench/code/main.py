"""
Lección: 32-minimal-agent-workbench
Fase: 14
Minimal agent workbench: tool registry,
memory, run loop, plan + execute,
verifier, structured output.
"""
from __future__ import annotations
import json
import time
import uuid


class Tool:
    def __init__(self, name, fn, description="", schema=None):
        self.name = name
        self.fn = fn
        self.description = description
        self.schema = schema or {}

    def call(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, tool):
        if not isinstance(tool, Tool):
            raise TypeError("expected Tool instance")
        self.tools[tool.name] = tool

    def get(self, name):
        return self.tools.get(name)

    def list(self):
        return list(self.tools.keys())

    def dispatch(self, name, *args, **kwargs):
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return tool.call(*args, **kwargs)


class Memory:
    def __init__(self):
        self.entries = []

    def add(self, role, content):
        e = {"id": str(uuid.uuid4()), "role": role, "content": content, "ts": time.time()}
        self.entries.append(e)
        return e

    def get(self, entry_id):
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        return None

    def all(self):
        return list(self.entries)

    def last(self, n=5):
        return list(self.entries[-n:])

    def clear(self):
        self.entries = []


class MinimalWorkbench:
    def __init__(self):
        self.tools = ToolRegistry()
        self.memory = Memory()
        self.max_steps = 10
        self.steps_taken = 0
        self.plan = []
        self.results = []

    def add_tool(self, tool):
        self.tools.register(tool)

    def set_plan(self, steps):
        self.plan = list(steps)

    def step(self, tool_name, *args, **kwargs):
        if self.steps_taken >= self.max_steps:
            raise RuntimeError("max_steps_exceeded")
        self.memory.add("assistant", f"Calling {tool_name}")
        result = self.tools.dispatch(tool_name, *args, **kwargs)
        self.memory.add("tool", f"{tool_name} -> {result}")
        self.results.append({"step": self.steps_taken, "tool": tool_name, "result": result})
        self.steps_taken += 1
        return result

    def run(self):
        for tool_name, args, kwargs in self.plan:
            self.step(tool_name, *args, **kwargs)
        return self.results

    def reset(self):
        self.memory.clear()
        self.steps_taken = 0
        self.results = []


def main() -> int:
    wb = MinimalWorkbench()
    wb.add_tool(Tool("echo", lambda x: x, description="Return input"))
    wb.set_plan([("echo", ("hello",), {})])
    print(wb.run())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())