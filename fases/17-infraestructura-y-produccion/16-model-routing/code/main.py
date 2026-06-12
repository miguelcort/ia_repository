"""
Lección: 16-model-routing
Fase: 17
Model routing: route requests to
different models based on complexity,
cost, latency, capability. Cascade,
specialist models, fall-back.
"""
from __future__ import annotations


class ModelRoute:
    def __init__(self, name, cost_per_1k, max_complexity, avg_latency):
        self.name = name
        self.cost_per_1k = cost_per_1k
        self.max_complexity = max_complexity
        self.avg_latency = avg_latency


class ModelRouter:
    def __init__(self, strategy="cheapest"):
        self.routes = []
        self.strategy = strategy

    def add_route(self, route):
        self.routes.append(route)

    def route(self, complexity, estimated_tokens=1000):
        candidates = [r for r in self.routes if r.max_complexity >= complexity]
        if not candidates:
            return None
        if self.strategy == "cheapest":
            return min(candidates, key=lambda r: r.cost_per_1k)
        if self.strategy == "fastest":
            return min(candidates, key=lambda r: r.avg_latency)
        if self.strategy == "best":
            return max(candidates, key=lambda r: r.max_complexity)
        return candidates[0]

    def estimate_cost(self, route, tokens):
        return route.cost_per_1k * tokens / 1000


def detect_complexity(prompt):
    """Rough complexity detection based on prompt length and keywords."""
    score = min(len(prompt) / 100, 5.0)
    keywords = ["analyze", "complex", "code", "math", "research"]
    for kw in keywords:
        if kw in prompt.lower():
            score += 1
    return score


def main() -> int:
    router = ModelRouter(strategy="cheapest")
    router.add_route(ModelRoute("haiku", 0.25, max_complexity=3, avg_latency=0.5))
    router.add_route(ModelRoute("sonnet", 3.0, max_complexity=8, avg_latency=2.0))
    router.add_route(ModelRoute("opus", 15.0, max_complexity=10, avg_latency=5.0))
    route = router.route(complexity=2)
    print(f"Route: {route.name}, est cost: ${router.estimate_cost(route, 1000):.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())