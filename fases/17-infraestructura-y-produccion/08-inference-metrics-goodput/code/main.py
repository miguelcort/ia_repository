"""
Lección: 08-inference-metrics-goodput
Fase: 17
Inference metrics: goodput (requests
meeting SLO), TTFT, TPOT, ITL, E2E
latency, throughput, utilization.
"""
from __future__ import annotations


class InferenceMetrics:
    def __init__(self):
        self.requests = []
        self.tokens_generated = 0
        self.duration_seconds = 0

    def record_request(self, ttft, tpot_per_token, total_tokens):
        self.requests.append({
            "ttft": ttft,
            "tpot": tpot_per_token,
            "tokens": total_tokens,
        })
        self.tokens_generated += total_tokens

    def total_requests(self):
        return len(self.requests)

    def avg_ttft(self):
        if not self.requests:
            return 0.0
        return sum(r["ttft"] for r in self.requests) / len(self.requests)

    def avg_tpot(self):
        if not self.requests:
            return 0.0
        return sum(r["tpot"] for r in self.requests) / len(self.requests)

    def tokens_per_second(self):
        if self.duration_seconds == 0:
            return 0.0
        return self.tokens_generated / self.duration_seconds

    def requests_per_second(self):
        if self.duration_seconds == 0:
            return 0.0
        return self.total_requests() / self.duration_seconds

    def goodput(self, slo_ttft, slo_tpot):
        """Fraction of requests meeting SLO."""
        if not self.requests:
            return 0.0
        good = sum(1 for r in self.requests if r["ttft"] <= slo_ttft and r["tpot"] <= slo_tpot)
        return good / len(self.requests)

    def e2e_latency_p99(self):
        if not self.requests:
            return 0.0
        e2e = sorted(r["ttft"] + r["tpot"] * r["tokens"] for r in self.requests)
        idx = int(len(e2e) * 0.99)
        return e2e[min(idx, len(e2e) - 1)]


def main() -> int:
    m = InferenceMetrics()
    m.record_request(0.1, 0.05, 100)
    m.record_request(0.2, 0.06, 150)
    m.duration_seconds = 1.0
    print(f"Avg TTFT: {m.avg_ttft():.3f}s")
    print(f"Goodput: {m.goodput(0.5, 0.1):.0%}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())