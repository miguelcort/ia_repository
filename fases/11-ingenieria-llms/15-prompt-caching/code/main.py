"""
Lección: 15-prompt-caching
Fase: 11
Prompt caching strategies: provider cache, semantic cache, response cache.
TTL, invalidation, monitoring.
"""
from __future__ import annotations
import sys
import numpy as np
import hashlib


def cache_key_from_prompt(prompt, model, temperature):
    """Compute cache key."""
    h = hashlib.md5(f"{model}:{temperature}:{prompt}".encode()).hexdigest()
    return h


def should_cache(prompt, n_chars=200):
    """Decide si prompt deberia estar cached.
    Heuristica: prompts largos + repetidos."""
    return len(prompt) > n_chars


def cache_hit_rate(requests, n_unique=10):
    """Mock: hit rate con n_unique cached."""
    if not requests:
        return 0.0
    return min(1.0, n_unique / len(requests))


def ttl_seconds(cache_type="response"):
    """Mock TTL por cache type."""
    ttls = {
        "response": 3600,  # 1h
        "semantic": 86400,  # 1d
        "provider": 300,  # 5min
    }
    return ttls.get(cache_type, 3600)


def main() -> int:
    key = cache_key_from_prompt("test", "gpt-4", 0)
    print(f"Key: {key[:16]}")
    h = cache_hit_rate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_unique=5)
    print(f"Hit rate: {h:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())