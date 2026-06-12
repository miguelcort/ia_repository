"""
Lección: 13-cost-governors
Fase: 15
Cost governors: budget caps, token limits,
model routing, rate limits, concurrency caps,
circuit breakers, soft + hard caps.
"""
from __future__ import annotations
import time


class CostGovernor:
    def __init__(self, daily_budget_usd=10.0, per_request_cap=0.5,
                 model_costs=None, window_seconds=86400):
        self.daily_budget_usd = daily_budget_usd
        self.per_request_cap = per_request_cap
        self.window_seconds = window_seconds
        self.model_costs = model_costs or {
            "gpt-4o": 5.0,
            "gpt-4o-mini": 0.15,
            "claude-3-5-sonnet": 3.0,
            "claude-3-haiku": 0.25,
        }
        self.spend_log = []

    def estimate_cost(self, model, input_tokens, output_tokens):
        per_million = self.model_costs.get(model, 5.0)
        return per_million * (input_tokens + output_tokens) / 1_000_000

    def allow(self, model, input_tokens, output_tokens):
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        if cost > self.per_request_cap:
            return False, cost, "exceeds_per_request_cap"
        total_spent = sum(c for _, c in self.spend_log)
        if total_spent + cost > self.daily_budget_usd:
            return False, cost, "exceeds_daily_budget"
        self.spend_log.append((time.time(), cost))
        return True, cost, "ok"

    def spent(self):
        return sum(c for _, c in self.spend_log)

    def remaining(self):
        return self.daily_budget_usd - self.spent()

    def reset(self):
        self.spend_log = []


def pick_cheapest_model(prompt_tokens, governor, candidates=None):
    """Pick the cheapest model that can handle the prompt."""
    candidates = candidates or ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "claude-3-haiku"]
    best = None
    best_cost = float("inf")
    for m in candidates:
        c = governor.estimate_cost(m, prompt_tokens, 200)
        if c < best_cost:
            best_cost = c
            best = m
    return best, best_cost


class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_seconds=60):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failures = 0
        self.opened_at = None

    def call(self, fn, *args, **kwargs):
        if self.opened_at is not None:
            if time.time() - self.opened_at < self.cooldown_seconds:
                raise RuntimeError("circuit_open")
            self.opened_at = None
            self.failures = 0
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.time()
            raise
        return result

    @property
    def is_open(self):
        if self.opened_at is None:
            return False
        return time.time() - self.opened_at < self.cooldown_seconds


def main() -> int:
    gov = CostGovernor(daily_budget_usd=1.0, per_request_cap=0.1)
    print(gov.allow("gpt-4o-mini", 1000, 200))
    print(gov.remaining())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())