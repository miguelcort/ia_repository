"""
Lección: 28-self-hosted-serving-selection
Fase: 17
Self-hosted serving selection: vLLM
vs TGI vs TensorRT-LLM vs LMDeploy
vs llama.cpp, hardware, scale, custom
needs, total cost of ownership.
"""
from __future__ import annotations


SERVING_OPTIONS = {
    "vllm": {
        "name": "vLLM",
        "language": "Python",
        "gpu_required": True,
        "throughput": "high",
        "ease_of_use": "medium",
        "year": 2023,
    },
    "tgi": {
        "name": "TGI",
        "language": "Rust/Python",
        "gpu_required": True,
        "throughput": "high",
        "ease_of_use": "high",
        "year": 2023,
    },
    "tensorrt_llm": {
        "name": "TensorRT-LLM",
        "language": "C++/Python",
        "gpu_required": True,
        "throughput": "very_high",
        "ease_of_use": "low",
        "year": 2024,
    },
    "lmdeploy": {
        "name": "LMDeploy",
        "language": "Python/C++",
        "gpu_required": True,
        "throughput": "high",
        "ease_of_use": "medium",
        "year": 2024,
    },
    "llama_cpp": {
        "name": "llama.cpp",
        "language": "C++",
        "gpu_required": False,
        "throughput": "medium",
        "ease_of_use": "medium",
        "year": 2023,
    },
}


def list_options():
    return list(SERVING_OPTIONS.keys())


def get_option(name):
    return SERVING_OPTIONS.get(name)


def by_throughput(min_throughput="medium"):
    order = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
    threshold = order.get(min_throughput, 0)
    return [k for k, v in SERVING_OPTIONS.items() if order.get(v["throughput"], 0) >= threshold]


def recommend(have_gpu=True, ease="high", min_throughput="medium"):
    """Recommend serving options based on constraints."""
    candidates = []
    for k, v in SERVING_OPTIONS.items():
        if v["gpu_required"] and not have_gpu:
            continue
        if v["ease_of_use"] not in ("high", "medium") and ease == "high":
            continue
        candidates.append(k)
    if not candidates:
        return None
    return candidates[0]


def main() -> int:
    print(f"Options: {list_options()}")
    print(f"No GPU: {recommend(have_gpu=False)}")
    print(f"GPU + high ease: {recommend(have_gpu=True, ease='high')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())