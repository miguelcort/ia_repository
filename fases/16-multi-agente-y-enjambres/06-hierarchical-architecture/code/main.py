"""
Lección: 06-hierarchical-architecture
Fase: 16
Hierarchical architecture: tree of
agents, delegation up/down, levels
of command, decomposition + synthesis.
"""
from __future__ import annotations
import time
import uuid


class HierarchicalNode:
    def __init__(self, agent_id=None, role="worker", level=0):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.role = role
        self.level = level
        self.parent = None
        self.children = []
        self.results = []

    def add_child(self, child):
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)
        return child

    def is_leaf(self):
        return len(self.children) == 0

    def delegate(self, task):
        """Delegate task to children and aggregate results."""
        if self.is_leaf():
            return self._execute(task)
        sub_results = [child.delegate(task) for child in self.children]
        return self._aggregate(sub_results)

    def _execute(self, task):
        result = f"leaf:{self.role}:{task.get('action', '?')}"
        self.results.append(result)
        return result

    def _aggregate(self, results):
        aggregated = f"agg:{self.role}:{','.join(results)}"
        self.results.append(aggregated)
        return aggregated

    def tree(self, depth=0):
        prefix = "  " * depth
        s = f"{prefix}- [{self.level}] {self.role}\n"
        for c in self.children:
            s += c.tree(depth + 1)
        return s


def build_tree():
    root = HierarchicalNode(role="manager", level=0)
    tech_lead = root.add_child(HierarchicalNode(role="tech_lead", level=1))
    product_lead = root.add_child(HierarchicalNode(role="product_lead", level=1))
    tech_lead.add_child(HierarchicalNode(role="developer", level=2))
    tech_lead.add_child(HierarchicalNode(role="tester", level=2))
    product_lead.add_child(HierarchicalNode(role="reviewer", level=2))
    return root


def main() -> int:
    root = build_tree()
    print(root.tree())
    print(root.delegate({"action": "ship"}))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())