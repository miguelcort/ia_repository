"""
Lección: 08-memory-blocks-sleep-time-compute
Fase: 14
Memory blocks + sleep-time compute (Letta 2024).
Periodic memory consolidation, background agents.
+Offline memory optimization. Block-based memory.
"""
from __future__ import annotations
import time
from collections import defaultdict


class MemoryBlock:
    """Named memory block (label + value + limit)."""
    def __init__(self, label, value="", limit=2000, description=""):
        self.label = label
        self.value = value
        self.limit = limit
        self.description = description
        self.updated_at = time.time()

    def update(self, new_value):
        """Update value, truncate to limit."""
        if len(new_value) > self.limit:
            new_value = new_value[:self.limit]
        self.value = new_value
        self.updated_at = time.time()

    def get(self):
        return self.value


class MemoryBlocksManager:
    """Manager for multiple memory blocks."""
    def __init__(self):
        self.blocks = {}

    def add_block(self, block):
        self.blocks[block.label] = block

    def get_block(self, label):
        return self.blocks.get(label)

    def update_block(self, label, new_value):
        block = self.blocks.get(label)
        if block:
            block.update(new_value)

    def all_blocks(self):
        return self.blocks

    def total_size(self):
        return sum(len(b.value) for b in self.blocks.values())


def sleep_time_consolidate(blocks_manager, consolidation_fn):
    """Sleep-time compute: periodic memory consolidation."""
    consolidated = {}
    for label, block in blocks_manager.all_blocks().items():
        old = block.get()
        new = consolidation_fn(old)
        if new != old:
            block.update(new)
            consolidated[label] = new
    return consolidated


def summarize_memory(value, max_length=500):
    """Mock consolidation: summarize to max_length."""
    if len(value) <= max_length:
        return value
    return value[:max_length] + "..."


def background_agent(blocks_manager, query):
    """Background agent que processa memory blocks."""
    results = {}
    for label, block in blocks_manager.all_blocks().items():
        if query.lower() in block.get().lower():
            results[label] = block.get()
    return results


def main() -> int:
    mgr = MemoryBlocksManager()
    mgr.add_block(MemoryBlock("persona", value="You are a helpful AI assistant.", limit=500))
    mgr.add_block(MemoryBlock("human", value="User's name is Alice.", limit=500))
    mgr.add_block(MemoryBlock("facts", value="The sky is blue. " * 50, limit=300))
    print(f"Total size: {mgr.total_size()}")
    consolidated = sleep_time_consolidate(mgr, summarize_memory)
    print(f"Consolidated: {list(consolidated.keys())}")
    bg = background_agent(mgr, "Alice")
    print(f"BG results: {list(bg.keys())}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())