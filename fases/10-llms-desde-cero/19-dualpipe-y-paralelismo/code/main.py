"""
Lección: 19-dualpipe-y-paralelismo
Fase: 10
DualPipe (DeepSeek 2025): bidirectional pipeline parallelism, 2x overlap compute/comm.
Pipeline schedules: 1F1B, Interleaved, ZB-H1, DualPipe.
"""
from __future__ import annotations
import sys
import numpy as np


def pipeline_bubble_fraction(n_microbatches, n_stages):
    """Pipeline bubble: (n_stages - 1) / n_microbatches.
    Tipico: 1F1B bubble = 5-10% con n_microbatches >> n_stages.
    """
    if n_microbatches == 0:
        return 1.0
    return max(0, (n_stages - 1) / n_microbatches)


def interleaved_bubble(n_microbatches, n_stages, n_chunks=2):
    """Interleaved (PipeDream): cada stage tiene n_chunks sub-stages.
    Bubble simplified = (n_stages * (n_chunks - 1)) / (n_microbatches + n_stages - 1) (top term).
    """
    if n_microbatches == 0:
        return 1.0
    return max(0, n_stages * (n_chunks - 1) / (n_microbatches + n_stages - 1))


def dualpipe_bubble(n_microbatches, n_stages):
    """DualPipe: bidirectional pipeline. Bubble ~ 0 if enough microbatches.
    """
    # 2x overlap forward y backward
    return pipeline_bubble_fraction(n_microbatches * 2, n_stages * 2)


def parallel_efficiency(compute_time, comm_time, num_workers):
    """Efficiency = compute / (compute + comm) * num_workers.
    """
    if compute_time + comm_time == 0:
        return 1.0
    return compute_time / (compute_time + comm_time) * num_workers / num_workers


def pipeline_schedules():
    return {
        "GPipe": "All forward, then all backward. +bubble.",
        "1F1B (one-forward-one-backward)": "Standard. Bubble = (S-1)/M.",
        "Interleaved (PipeDream)": "Multiple chunks per stage. -bubble.",
        "ZB-H1 / ZB-H2 (Zero Bubble)": "Sync sin bubble en 1F1B.",
        "DualPipe (DeepSeek-V3)": "Bidirectional, 2x overlap compute/comm.",
        "FlexiblePipe": "Pipeline paralelo, async.",
    }


def main() -> int:
    # Bubble comparisons
    n_mb, n_s = 64, 8
    print(f"1F1B bubble (M=64, S=8): {pipeline_bubble_fraction(n_mb, n_s):.3f}")
    print(f"Interleaved (M=64, S=8, V=2): {interleaved_bubble(n_mb, n_s):.3f}")
    print(f"DualPipe (M=64, S=8): {dualpipe_bubble(n_mb, n_s):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())