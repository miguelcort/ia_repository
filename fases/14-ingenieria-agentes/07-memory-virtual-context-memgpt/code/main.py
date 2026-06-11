"""
Lección: 07-memory-virtual-context-memgpt
Fase: 14
MemGPT (Packer 2023): virtual context management para LLMs.
Hierarchical memory: core, archival, recall. Page in/out.
+Infinite context via OS-like memory management.
"""
from __future__ import annotations
import time


class VirtualContextMemory:
    """Mock MemGPT-style virtual context memory."""
    def __init__(self, core_size=10, archival_size=1000, recall_size=20):
        self.core = []  # in-context (limited)
        self.core_size = core_size
        self.archival = []  # out-of-context (unlimited)
        self.archival_size = archival_size
        self.recall = []  # recent (rolling)
        self.recall_size = recall_size
        self.page_in_count = 0
        self.page_out_count = 0

    def add_to_core(self, item):
        """Add to core memory (in-context)."""
        if len(self.core) >= self.core_size:
            # page out oldest to archival
            oldest = self.core.pop(0)
            self.archival.append(oldest)
            self.page_out_count += 1
        self.core.append(item)

    def page_in_from_archival(self, query, n=1):
        """Page in n items matching query from archival to core."""
        results = []
        for item in self.archival:
            if query.lower() in str(item).lower():
                results.append(item)
                if len(results) >= n:
                    break
        for r in results:
            self.archival.remove(r)
            self.page_in_count += 1
            self.add_to_core(r)
        return results

    def search_archival(self, query, n=5):
        """Search archival without paging in."""
        results = []
        for item in self.archival:
            if query.lower() in str(item).lower():
                results.append(item)
                if len(results) >= n:
                    break
        return results

    def add_to_recall(self, item):
        """Add to recall (rolling window)."""
        self.recall.append(item)
        if len(self.recall) > self.recall_size:
            self.recall.pop(0)

    def get_core(self):
        return self.core

    def get_recall(self):
        return self.recall

    def get_archival_size(self):
        return len(self.archival)


def main() -> int:
    mem = VirtualContextMemory(core_size=3, recall_size=5)
    for i in range(5):
        mem.add_to_core(f"item {i}")
    print(f"Core: {mem.get_core()}")
    print(f"Archival size: {mem.get_archival_size()}")
    print(f"Page in: {mem.page_in_count}, out: {mem.page_out_count}")
    results = mem.search_archival("item 1")
    print(f"Search: {results}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())