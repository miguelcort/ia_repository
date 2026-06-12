"""
Lección: 10-cold-start-mitigation
Fase: 17
Cold start mitigation: keep models warm,
predictive pre-warm, snapshot caching,
lazy loading, JIT compilation cache.
"""
from __future__ import annotations
import time


class ModelPool:
    def __init__(self, max_size=10, idle_timeout=300):
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        self.loaded = {}

    def warm(self, model_name, load_fn):
        if model_name in self.loaded:
            return False
        if len(self.loaded) >= self.max_size:
            self._evict_idle()
        self.loaded[model_name] = {
            "model": load_fn(),
            "last_used": time.time(),
        }
        return True

    def get(self, model_name):
        if model_name not in self.loaded:
            return None
        self.loaded[model_name]["last_used"] = time.time()
        return self.loaded[model_name]["model"]

    def _evict_idle(self):
        now = time.time()
        for name in list(self.loaded.keys()):
            if now - self.loaded[name]["last_used"] > self.idle_timeout:
                del self.loaded[name]
                if len(self.loaded) < self.max_size:
                    return

    def size(self):
        return len(self.loaded)


class PredictiveWarmer:
    def __init__(self, pool):
        self.pool = pool
        self.history = []

    def record_request(self, model_name):
        self.history.append({"model": model_name, "ts": time.time()})

    def predict_next_models(self, top_k=2):
        """Return top-k most-requested models."""
        counts = {}
        for h in self.history:
            counts[h["model"]] = counts.get(h["model"], 0) + 1
        sorted_models = sorted(counts.items(), key=lambda kv: -kv[1])
        return [m for m, _ in sorted_models[:top_k]]

    def warm_predicted(self, load_fn):
        for model_name in self.predict_next_models():
            self.pool.warm(model_name, lambda: load_fn(model_name))


def main() -> int:
    pool = ModelPool(max_size=3, idle_timeout=600)
    pool.warm("m1", lambda: "model-1")
    print(f"Pool size: {pool.size()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())