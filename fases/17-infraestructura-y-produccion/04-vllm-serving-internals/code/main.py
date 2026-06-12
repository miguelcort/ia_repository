"""
Lección: 04-vllm-serving-internals
Fase: 17
vLLM serving internals: PagedAttention,
continuous batching, prefix caching,
tensor parallelism, throughput.
"""
from __future__ import annotations


class PagedAttention:
    def __init__(self, block_size=16, num_blocks=100):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks = [None] * num_blocks
        self.allocations = {}

    def allocate(self, seq_id, num_tokens):
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        free = [i for i, b in enumerate(self.blocks) if b is None]
        if len(free) < blocks_needed:
            return False
        allocated = free[:blocks_needed]
        for idx in allocated:
            self.blocks[idx] = (seq_id, idx)
        self.allocations[seq_id] = allocated
        return True

    def free(self, seq_id):
        if seq_id in self.allocations:
            for idx in self.allocations[seq_id]:
                self.blocks[idx] = None
            del self.allocations[seq_id]

    def memory_used(self):
        return sum(1 for b in self.blocks if b is not None) * self.block_size

    def memory_total(self):
        return self.num_blocks * self.block_size


class ContinuousBatcher:
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.active = []

    def add(self, request):
        if len(self.active) >= self.max_batch_size:
            return False
        self.active.append(request)
        return True

    def step(self):
        for r in self.active:
            r["generated"] = r.get("generated", 0) + 1
        self.active = [r for r in self.active if r["generated"] < r["max_tokens"]]
        return self.active

    def active_count(self):
        return len(self.active)


def prefix_cache_hit(prompt, cache):
    """Return cached prefix length."""
    match_len = 0
    for cached_prompt, cached_state in cache.items():
        if prompt.startswith(cached_prompt):
            if len(cached_prompt) > match_len:
                match_len = len(cached_prompt)
    return match_len


def main() -> int:
    pa = PagedAttention(block_size=16, num_blocks=10)
    pa.allocate("seq1", 32)
    print(f"Memory: {pa.memory_used()}/{pa.memory_total()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())