"""
Lección: 21-llm-routing-layer
Fase: 13
LLM routing: intelligent routing entre modelos.
Cost, latency, capability, load balancing, fallback.
LiteLLM, Portkey, OpenRouter, semantic router.
"""
from __future__ import annotations


def route_by_cost(prompt, models):
    """Route al model mas barato disponible."""
    sorted_models = sorted(models, key=lambda m: m["cost_per_1k"])
    return sorted_models[0]


def route_by_latency(prompt, models, latency_history):
    """Route al model con menor latency promedio."""
    avg_lat = {m["name"]: sum(latency_history.get(m["name"], [0.1])) / max(1, len(latency_history.get(m["name"], [0.1]))) for m in models}
    return min(models, key=lambda m: avg_lat[m["name"]])


def route_by_capability(prompt, models, required_capability):
    """Route al model con capability requerida."""
    candidates = [m for m in models if required_capability in m.get("capabilities", [])]
    if not candidates:
        return None
    return candidates[0]


def fallback_chain(prompt, models, primary_fn, fallback_fn=None):
    """Try primary, fallback on error."""
    for m in models:
        try:
            return primary_fn(m, prompt)
        except Exception:
            if fallback_fn:
                return fallback_fn(m, prompt)
            continue
    return None


def load_balance_round_robin(prompt, models, n_requests=10):
    """Round-robin load balancing."""
    counts = {m["name"]: 0 for m in models}
    for i in range(n_requests):
        m = models[i % len(models)]
        counts[m["name"]] += 1
    return counts


def semantic_route(prompt, embeddings, model_embeddings, threshold=0.7):
    """Route basado en semantic similarity entre prompt y model descriptions."""
    sims = []
    for i, me in enumerate(model_embeddings):
        # dot product (list)
        s = sum(a * b for a, b in zip(embeddings, me)) / (sum(embeddings) + 1e-8)
        sims.append((i, float(s)))
    sims.sort(key=lambda x: x[1], reverse=True)
    if sims[0][1] >= threshold:
        return sims[0][0]
    return None  # no match


def main() -> int:
    models = [
        {"name": "gpt-4o", "cost_per_1k": 5.0, "capabilities": ["vision", "tools"]},
        {"name": "gpt-4o-mini", "cost_per_1k": 0.15, "capabilities": ["vision"]},
        {"name": "claude-3-5-sonnet", "cost_per_1k": 3.0, "capabilities": ["vision", "tools", "code"]},
    ]
    cheapest = route_by_cost("test", models)
    print(f"Cheapest: {cheapest['name']}")
    code_model = route_by_capability("test", models, "code")
    print(f"Code-capable: {code_model['name']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())