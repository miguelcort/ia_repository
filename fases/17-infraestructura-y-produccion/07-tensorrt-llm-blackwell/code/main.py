"""
Lección: 07-tensorrt-llm-blackwell
Fase: 17
TensorRT-LLM on Blackwell: NVIDIA
optimization, kernel fusion, FP4/FP8,
in-flight batching, optimized for
Hopper/Blackwell GPUs.
"""
from __future__ import annotations


NVIDIA_GPUS = {
    "A100": {
        "name": "A100",
        "architecture": "Ampere",
        "memory_gb": 80,
        "fp8_tflops": 0,
        "tensor_cores": 432,
    },
    "H100": {
        "name": "H100",
        "architecture": "Hopper",
        "memory_gb": 80,
        "fp8_tflops": 1979,
        "tensor_cores": 528,
    },
    "H200": {
        "name": "H200",
        "architecture": "Hopper",
        "memory_gb": 141,
        "fp8_tflops": 1979,
        "tensor_cores": 528,
    },
    "B100": {
        "name": "B100",
        "architecture": "Blackwell",
        "memory_gb": 192,
        "fp8_tflops": 3600,
        "tensor_cores": 864,
    },
    "B200": {
        "name": "B200",
        "architecture": "Blackwell",
        "memory_gb": 192,
        "fp8_tflops": 4500,
        "tensor_cores": 1080,
    },
}


def list_gpus():
    return list(NVIDIA_GPUS.keys())


def get_gpu(name):
    return NVIDIA_GPUS.get(name)


def best_for_throughput():
    """Return GPU with highest fp8_tflops."""
    return max(NVIDIA_GPUS.items(), key=lambda kv: kv[1]["fp8_tflops"])


def best_for_memory():
    """Return GPU with highest memory_gb."""
    return max(NVIDIA_GPUS.items(), key=lambda kv: kv[1]["memory_gb"])


def can_run_model(gpu_name, model_size_b):
    """Estimate if a GPU can run a model (simplified)."""
    gpu = NVIDIA_GPUS.get(gpu_name)
    if not gpu:
        return False
    bytes_per_param = 2 if "H" in gpu["architecture"] or "B" in gpu["architecture"] else 4
    required_gb = model_size_b * bytes_per_param
    return gpu["memory_gb"] >= required_gb


def main() -> int:
    print(f"GPUs: {list_gpus()}")
    print(f"Best throughput: {best_for_throughput()[0]}")
    print(f"Best memory: {best_for_memory()[0]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())