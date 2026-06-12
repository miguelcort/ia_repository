"""
Lección: 12-edge-inference
Fase: 17
Edge inference: on-device LLM, mobile
+ IoT, model compression, latency
requirements, fallback to cloud,
model partitioning.
"""
from __future__ import annotations


class EdgeDevice:
    def __init__(self, name, memory_mb=4096, compute_score=10.0,
                 supports_quantization=True):
        self.name = name
        self.memory_mb = memory_mb
        self.compute_score = compute_score
        self.supports_quantization = supports_quantization
        self.running_model = None

    def can_run(self, model_size_mb, requires_quantization=False):
        if requires_quantization and not self.supports_quantization:
            return False
        return self.memory_mb >= model_size_mb

    def load(self, model_name, model_size_mb):
        if not self.can_run(model_size_mb):
            return False
        self.running_model = model_name
        return True

    def unload(self):
        self.running_model = None


class EdgeRouter:
    def __init__(self):
        self.devices = []
        self.cloud_endpoint = None

    def register_device(self, device):
        self.devices.append(device)

    def set_cloud(self, endpoint):
        self.cloud_endpoint = endpoint

    def route(self, model_name, model_size_mb, requires_quantization=False):
        for device in self.devices:
            if device.can_run(model_size_mb, requires_quantization):
                if device.load(model_name, model_size_mb):
                    return ("edge", device.name)
        if self.cloud_endpoint:
            return ("cloud", self.cloud_endpoint)
        return (None, "no_capacity")


def estimate_latency(model_size_mb, compute_score, tokens=100):
    """Rough latency estimate: model_size / compute."""
    return (model_size_mb / max(compute_score, 0.1)) * 0.01 * tokens


def main() -> int:
    phone = EdgeDevice("phone", memory_mb=8192, compute_score=15.0)
    router = EdgeRouter()
    router.register_device(phone)
    router.set_cloud("https://api.example.com")
    print(router.route("llama-3-8b-int4", 4500, requires_quantization=True))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())