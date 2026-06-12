"""
Lección: 18-vllm-production-stack-lmcache
Fase: 17
vLLM production stack with LMCache:
distributed KV cache, prefix sharing
across instances, GPU memory tier,
cache eviction.
"""
from __future__ import annotations
import time


class LMCache:
    def __init__(self, max_size_mb=1024):
        self.max_size_mb = max_size_mb
        self.entries = {}
        self.access_log = []

    def put(self, key, value, size_mb):
        if self.size_mb() + size_mb > self.max_size_mb:
            self._evict(size_mb)
        self.entries[key] = {
            "value": value,
            "size_mb": size_mb,
            "last_access": time.time(),
        }
        self.access_log.append({"op": "put", "key": key, "size": size_mb})

    def get(self, key):
        self.access_log.append({"op": "get", "key": key})
        if key not in self.entries:
            return None
        self.entries[key]["last_access"] = time.time()
        return self.entries[key]["value"]

    def _evict(self, needed_mb):
        sorted_entries = sorted(
            self.entries.items(),
            key=lambda kv: kv[1]["last_access"],
        )
        for k, v in sorted_entries:
            if self.size_mb() + needed_mb <= self.max_size_mb:
                return
            self.access_log.append({"op": "evict", "key": k})
            del self.entries[k]

    def size_mb(self):
        return sum(e["size_mb"] for e in self.entries.values())

    def count(self):
        return len(self.entries)

    def hit_rate(self):
        if not self.access_log:
            return 0.0
        gets = [e for e in self.access_log if e["op"] == "get"]
        if not gets:
            return 0.0
        hits = sum(1 for e in gets if e["key"] in self.entries)
        return hits / len(gets)


def main() -> int:
    cache = LMCache(max_size_mb=100)
    cache.put("k1", "v1", 50)
    cache.put("k2", "v2", 60)
    print(f"Size: {cache.size_mb()}MB, count: {cache.count()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())