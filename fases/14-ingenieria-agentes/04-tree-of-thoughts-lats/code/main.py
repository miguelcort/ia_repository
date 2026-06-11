"""
Lección: 04-tree-of-thoughts-lats
Fase: 14
Tree of Thoughts (ToT, Yao 2023) y LATS (Language Agent
Tree Search, Zhou 2023). BFS/DFS + MCTS para exploration.
+Reasoning via tree search.
"""
from __future__ import annotations
import math
import heapq


class TreeNode:
    """Node en ToT/LATS tree."""
    def __init__(self, state, parent=None, action=None, value=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.value = value
        self.children = []
        self.visits = 0

    def add_child(self, child):
        self.children.append(child)
        return child

    def is_leaf(self):
        return len(self.children) == 0

    def path(self):
        """Path from root."""
        p = []
        n = self
        while n is not None:
            p.append(n)
            n = n.parent
        return list(reversed(p))


def bfs_expand(root, expand_fn, evaluate_fn, max_depth=3):
    """BFS expansion of tree."""
    current_level = [root]
    for _ in range(max_depth):
        next_level = []
        for node in current_level:
            if evaluate_fn(node.state) == "goal":
                return node
            for action in expand_fn(node.state):
                child = TreeNode(state=action["next_state"], parent=node, action=action)
                child.value = evaluate_fn(child.state)
                node.add_child(child)
                next_level.append(child)
        current_level = next_level
    return None


def ucb_select(node, c=1.414):
    """UCB1 selection for MCTS."""
    if node.visits == 0:
        return float("inf")
    return node.value / node.visits + c * math.sqrt(math.log(max(1, node.parent.visits)) / node.visits) if node.parent else node.value


def mcts(root, expand_fn, evaluate_fn, simulate_fn, n_iterations=100, c=1.414):
    """MCTS con UCB1."""
    for _ in range(n_iterations):
        # selection
        node = root
        while not node.is_leaf():
            node = max(node.children, key=lambda child: ucb_select(child, c=c))
        # expansion
        for action in expand_fn(node.state):
            child = TreeNode(state=action["next_state"], parent=node, action=action)
            child.value = evaluate_fn(child.state)
            node.add_child(child)
        # simulation
        reward = simulate_fn(node.state)
        # backprop
        n = node
        while n is not None:
            n.visits += 1
            n.value += reward
            n = n.parent
    # return best child
    if not root.children:
        return root
    return max(root.children, key=lambda c: c.value / max(1, c.visits))


def main() -> int:
    root = TreeNode(state="start", value=0.0)
    root.visits = 1
    expand_fn = lambda s: [{"next_state": s + "_a"}, {"next_state": s + "_b"}]
    evaluate_fn = lambda s: 1.0 if "a" in s else 0.5
    best = mcts(root, expand_fn, evaluate_fn, evaluate_fn, n_iterations=20)
    print(f"Best: {best.state}, value: {best.value / max(1, best.visits):.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())