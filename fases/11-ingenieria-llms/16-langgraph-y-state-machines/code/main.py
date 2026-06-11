"""
Lección: 16-langgraph-y-state-machines
Fase: 11
LangGraph: stateful, multi-actor, agentic workflows.
Nodes, edges, conditional edges, state machines.
"""
from __future__ import annotations
import sys
import numpy as np


def graph_node(state):
    """Mock graph node: procesa state y retorna update."""
    return {"messages": state.get("messages", []) + ["processed"]}


def graph_edge(from_node, to_node, condition=None):
    """Mock graph edge."""
    if condition is None:
        return True  # unconditional
    return bool(condition())


def graph_state_init():
    """Initial state para el graph."""
    return {"messages": [], "step": 0}


def graph_step(state, transition_fn, max_steps=10):
    """Run graph hasta max_steps o done."""
    for i in range(max_steps):
        state = transition_fn(state)
        if state.get("done"):
            break
    return state


def build_simple_graph():
    """Build a simple 3-node graph: input -> process -> output."""
    nodes = {
        "input": lambda s: {**s, "step": 1},
        "process": lambda s: {**s, "step": 2, "result": "done"},
        "output": lambda s: {**s, "done": True, "step": 3},
    }
    edges = [
        ("input", "process", None),
        ("process", "output", None),
    ]
    return nodes, edges


def main() -> int:
    nodes, edges = build_simple_graph()
    state = graph_state_init()
    state = nodes["input"](state)
    state = nodes["process"](state)
    state = nodes["output"](state)
    print(f"Final state: {state}")
    return 0


if __name__ == "__main__":
    sys.exit(main())