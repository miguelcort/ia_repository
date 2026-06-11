"""
Lección: 13-langgraph-stateful-graphs
Fase: 14
LangGraph (LangChain 2024): stateful graphs para agents.
Nodes, edges, cycles, state, conditional edges, persistence.
+Production agent framework.
"""
from __future__ import annotations
from collections import defaultdict


class State:
    """Mock LangGraph state."""
    def __init__(self, initial=None):
        self.data = initial or {}
        self.history = []

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.history.append({key: value})

    def update(self, updates):
        self.data.update(updates)
        self.history.append(updates)

    def snapshot(self):
        return dict(self.data)


class Node:
    """Node in graph."""
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def run(self, state):
        return self.fn(state)


class Edge:
    """Edge connecting nodes."""
    def __init__(self, src, dst, condition=None):
        self.src = src
        self.dst = dst
        self.condition = condition  # callable(state) -> bool


class Graph:
    """Mock LangGraph."""
    def __init__(self):
        self.nodes = {}
        self.edges = defaultdict(list)  # src -> [Edge]
        self.entry = None

    def add_node(self, name, fn):
        self.nodes[name] = Node(name, fn)

    def add_edge(self, src, dst, condition=None):
        self.edges[src].append(Edge(src, dst, condition))

    def set_entry(self, name):
        self.entry = name

    def step(self, current, state):
        """Execute current node + determine next via edges."""
        if current not in self.nodes:
            return None
        result = self.nodes[current].run(state)
        # find next
        for edge in self.edges[current]:
            if edge.condition is None or edge.condition(state):
                return edge.dst
        return None  # no edge

    def run(self, state, max_steps=20):
        """Run graph from entry until no more edges or max_steps."""
        current = self.entry
        steps = 0
        while current is not None and steps < max_steps:
            current = self.step(current, state)
            steps += 1
        return state


def conditional_edge(state):
    """Conditional: go to 'end' if state has 'done' key."""
    return "done" in state.data


def main() -> int:
    g = Graph()
    g.set_entry("start")
    g.add_node("start", lambda s: s.update({"step": 1}))
    g.add_node("process", lambda s: s.update({"step": s.get("step", 0) + 1}))
    g.add_node("end", lambda s: s.update({"done": True}))
    g.add_edge("start", "process")
    g.add_edge("process", "end", condition=conditional_edge)
    state = State()
    g.run(state, max_steps=10)
    print(f"Final: {state.snapshot()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())