"""
Lección: 11-caching-y-costo
Fase: 11
LLM caching y cost optimization: prompt cache, response cache, batching,
KV cache. Cost monitoring.
"""
from __future__ import annotations
import sys
import numpy as np


def estimate_cost(n_tokens_in, n_tokens_out, cost_per_1k_in, cost_per_1k_out):
    """Estima cost de una llamada LLM."""
    return (n_tokens_in / 1000) * cost_per_1k_in + (n_tokens_out / 1000) * cost_per_1k_out


def cache_key(prompt, model, temperature):
    """Simple cache key."""
    return f"{model}:{temperature}:{hash(prompt)}"


def response_cache(cache, prompt, model="gpt-4", temperature=0):
    """Mock response cache: hit o miss."""
    key = cache_key(prompt, model, temperature)
    return cache.get(key)


def cache_with_ttl(cache, max_size=1000, ttl_seconds=3600):
    """Mock: simulate LRU + TTL cache."""
    # En production: redis, memcached
    return cache


def prompt_cache_savings(requests, cache_hit_rate, tokens_in=500, cost_per_1k=0.01):
    """Calcular savings con prompt cache."""
    hits = int(requests * cache_hit_rate)
    miss_cost = (requests - hits) * tokens_in / 1000 * cost_per_1k
    hit_cost = hits * tokens_in / 1000 * cost_per_1k * 0.1  # cache read 10x
    return miss_cost + hit_cost


def main() -> int:
    cost = estimate_cost(1000, 500, 0.03, 0.06)  # GPT-4 pricing
    print(f"Cost: ${cost:.4f}")
    saved = prompt_cache_savings(1000, cache_hit_rate=0.5)
    full = prompt_cache_savings(1000, cache_hit_rate=0.0)
    print(f"With 50% cache hit: ${saved:.2f}, vs ${full:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())