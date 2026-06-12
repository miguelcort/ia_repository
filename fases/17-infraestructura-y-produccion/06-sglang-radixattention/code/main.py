"""
Lección: 06-sglang-radixattention
Fase: 17
SGLang RadixAttention: tree-based
prefix cache for LLM serving, fast
lookup, structured generation, share
prefixes across requests.
"""
from __future__ import annotations
import time


class RadixNode:
    def __init__(self, key=None):
        self.key = key
        self.children = {}
        self.last_access = 0
        self.value = None
        self.is_leaf = True

    def touch(self):
        self.last_access = time.time()


class RadixTree:
    def __init__(self):
        self.root = RadixNode()
        self.size = 0

    def insert(self, tokens, value=None):
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = RadixNode(key=token)
                node.is_leaf = False
            node = node.children[token]
            node.touch()
        node.value = value
        node.is_leaf = True
        self.size += 1

    def lookup(self, tokens):
        node = self.root
        matched = []
        for token in tokens:
            if token in node.children:
                node = node.children[token]
                node.touch()
                matched.append(token)
            else:
                break
        return matched

    def evict_lru(self, current_time=None):
        if current_time is None:
            current_time = time.time()
        evicted = 0
        for path in self._walk_leaves(self.root):
            for n in path:
                if n.last_access < current_time - 3600:
                    n.value = None
                    evicted += 1
        return evicted

    def _walk_leaves(self, node, path=None):
        if path is None:
            path = []
        path.append(node)
        if node.is_leaf:
            yield path
        for child in node.children.values():
            yield from self._walk_leaves(child, path)
        path.pop()


def main() -> int:
    tree = RadixTree()
    tree.insert(["hello", "world"], value="hw_state")
    matched = tree.lookup(["hello", "world", "foo"])
    print(f"Matched: {matched}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())