"""
Lección: 05-scaling-y-distribuido
Fase: 10
Data parallel (DDP), tensor parallel (TP), pipeline parallel (PP), ZeRO/FSDP.
3D parallelism, FSDP, sequence parallel, expert parallel.
"""
from __future__ import annotations
import sys
import numpy as np


def data_parallel_size(n_gpus, batch_per_gpu):
    """Effective batch = n_gpus * batch_per_gpu."""
    return n_gpus * batch_per_gpu


def tensor_parallel_layers(n_layers, tp_degree):
    """En TP, cada GPU tiene n_layers / tp_degree layers.
    Asumimos uniform split.
    """
    return n_layers // tp_degree


def pipeline_parallel_stages(n_layers, pp_degree):
    """En PP, divide layers en pp_degree stages.
    Cada stage tiene n_layers / pp_degree layers.
    """
    return n_layers // pp_degree


def zero_optimizer_states(optimizer_states_gb, n_gpus):
    """ZeRO-1: sharding de optimizer states across n_gpus.
    Memory reduction: divide by n_gpus.
    """
    return optimizer_states_gb / n_gpus


class FSDPConfig:
    """Mock de Fully Sharded Data Parallel config."""

    def __init__(self, n_gpus, model_params_gb, optimizer_states_gb, grad_gb, sharding_strategy="full"):
        self.n_gpus = n_gpus
        self.model_params_gb = model_params_gb
        self.optimizer_states_gb = optimizer_states_gb
        self.grad_gb = grad_gb
        self.sharding_strategy = sharding_strategy

    def memory_per_gpu(self):
        """Memory per GPU despues de FSDP sharding."""
        if self.sharding_strategy == "full":
            return (self.model_params_gb + self.optimizer_states_gb + self.grad_gb) / self.n_gpus
        elif self.sharding_strategy == "grads_only":
            return self.model_params_gb + self.optimizer_states_gb / self.n_gpus + self.grad_gb / self.n_gpus
        elif self.sharding_strategy == "no_shard":
            return self.model_params_gb + self.optimizer_states_gb + self.grad_gb
        return 0.0


def zero_stages_summary():
    """ZeRO stages (DeepSpeed)."""
    return {
        "ZeRO-1": "Optimizer states sharded (4x memory reduction)",
        "ZeRO-2": "+ Gradients sharded (8x memory reduction)",
        "ZeRO-3": "+ Parameters sharded (linear memory reduction)",
        "FSDP": "PyTorch native equivalent, full sharding",
    }


def parallelism_3d(n_gpus, tp_degree, pp_degree):
    """3D parallelism: TP x PP x DP = n_gpus.
    DP = n_gpus / (TP * PP).
    """
    if tp_degree * pp_degree > n_gpus:
        return None
    return {
        "tp_degree": tp_degree,
        "pp_degree": pp_degree,
        "dp_degree": n_gpus // (tp_degree * pp_degree),
    }


def communication_overhead(n_gpus, model_size_gb, network_gbps=200):
    """All-reduce communication overhead en segundos (simplificado).
    network_gbps: interconnect speed (NVLink 200, IB 400).
    """
    # 2 * (n-1)/n * model_size / bandwidth
    if n_gpus <= 1:
        return 0.0
    return 2 * (n_gpus - 1) / n_gpus * model_size_gb * 8 / network_gbps


def main() -> int:
    print("=== FSDP memory (7B model, 7GB params, 14GB optim, 7GB grads) ===")
    config = FSDPConfig(n_gpus=8, model_params_gb=7, optimizer_states_gb=14, grad_gb=7)
    for strat in ["no_shard", "grads_only", "full"]:
        config.sharding_strategy = strat
        print(f"  {strat}: {config.memory_per_gpu():.2f} GB per GPU")
    # 3D parallelism
    print("\n=== 3D parallelism (64 GPUs) ===")
    p = parallelism_3d(64, tp_degree=8, pp_degree=4)
    print(f"  TP=8, PP=4, DP={p['dp_degree']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())