"""
Lección: 12-optimizacion-de-inferencia
Fase: 10
Inference optimization: KV cache, Flash Attention, paged attention, continuous
batching, prefix caching, tensor parallelism inference, chunked prefill.
"""
from __future__ import annotations
import sys
import numpy as np


def kv_cache_memory(n_layers, n_heads, d_k, seq_len, batch_size, num_bytes=2):
    """KV cache memory: 2 * n_layers * n_heads * seq_len * d_k * num_bytes per batch.
    num_bytes: 2 (fp16), 1 (int8), 0.5 (int4).
    """
    return 2 * n_layers * n_heads * seq_len * d_k * num_bytes * batch_size


def paged_attention_blocks(total_tokens, block_size=16):
    """Paged attention: divide tokens en blocks. Memory wasted ~ block_size / 2."""
    n_blocks = (total_tokens + block_size - 1) // block_size
    return n_blocks


def memory_waste_paged(total_tokens, block_size=16):
    """Memory waste: hasta block_size - 1 per sequence."""
    n_sequences = 4  # mock
    return block_size * n_sequences


def continuous_batching_throughput(n_requests, max_batch_size, tokens_per_sec):
    """Continuous batching: throughput = max_batch_size * tokens_per_sec (steady state)."""
    if n_requests < max_batch_size:
        return n_requests * tokens_per_sec
    return max_batch_size * tokens_per_sec


def prefix_cache_hit_rate(prefix_lengths, n_requests):
    """Prefix caching: shared prefix entre requests. Hit rate = shared / total.
    """
    total = sum(prefix_lengths)
    shared = sum(prefix_lengths) - max(prefix_lengths) if prefix_lengths else 0
    return shared / total if total else 0.0


def chunked_prefill_schedule(seq_lens, chunk_size=512):
    """Chunked prefill: process long sequences en chunks, mix con decode.
    """
    n_chunks = sum((l + chunk_size - 1) // chunk_size for l in seq_lens)
    return n_chunks


def inference_optimizations():
    return {
        "KV cache": "Reusa K, V pasados, O(n) por step",
        "Flash Attention": "O(n) memoria, 2-4x speedup",
        "Paged attention": "vLLM, bloques no contiguos",
        "Continuous batching": "Mix requests, +throughput",
        "Prefix caching": "Shared prefix entre requests",
        "Speculative decoding": "Draft chico + target grande",
        "Tensor parallel": "Split model entre GPUs",
        "Chunked prefill": "Long sequences en chunks",
    }


def main() -> int:
    # Llama 2 70B KV cache
    # 80 layers, 8 KV heads (GQA), d_k=128, bf16
    mem = kv_cache_memory(n_layers=80, n_heads=8, d_k=128, seq_len=4096, batch_size=1)
    print(f"Llama 2 70B KV cache (bs=1, seq=4096): {mem / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())