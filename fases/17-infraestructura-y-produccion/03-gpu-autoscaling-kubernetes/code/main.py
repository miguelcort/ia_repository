"""
Lección: 03-gpu-autoscaling-kubernetes
Fase: 17
GPU autoscaling in Kubernetes: HPA for
GPUs, KEDA, Karpenter, custom metrics,
node pools, spot vs on-demand.
"""
from __future__ import annotations
import time


class NodePool:
    def __init__(self, name, gpu_type, count, hourly_cost):
        self.name = name
        self.gpu_type = gpu_type
        self.count = count
        self.hourly_cost = hourly_cost
        self.allocated = 0

    def available(self):
        return self.count - self.allocated

    def allocate(self, n=1):
        if self.allocated + n > self.count:
            return False
        self.allocated += n
        return True


class Cluster:
    def __init__(self):
        self.pools = []

    def add_pool(self, pool):
        self.pools.append(pool)

    def total_gpus(self):
        return sum(p.count for p in self.pools)

    def available_gpus(self):
        return sum(p.available() for p in self.pools)

    def allocate(self, n=1):
        for p in self.pools:
            if p.allocate(n):
                return p.name
        return None

    def scale_up(self, pool_name, delta):
        for p in self.pools:
            if p.name == pool_name:
                p.count += delta
                return p.count
        return None


def decide_scale(metric_value, threshold_up=70, threshold_down=30):
    """Return action: 'up', 'down', or 'stable'."""
    if metric_value > threshold_up:
        return "up"
    if metric_value < threshold_down:
        return "down"
    return "stable"


def cost_per_hour(cluster):
    return sum(p.count * p.hourly_cost for p in cluster.pools)


def main() -> int:
    c = Cluster()
    c.add_pool(NodePool("a100", "A100", 4, 3.0))
    c.add_pool(NodePool("h100", "H100", 2, 8.0))
    print(f"GPUs: {c.total_gpus()}, ${cost_per_hour(c)}/h")
    print(f"Scale: {decide_scale(85)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())