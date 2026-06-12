"""
Lección: 21-ab-testing-llm-features
Fase: 17
A/B testing LLM features: control vs
variant, statistical significance,
sample size, conversion, A/B/n,
multi-variant.
"""
from __future__ import annotations
import math


class ABTest:
    def __init__(self, name, control_name, variants):
        self.name = name
        self.control_name = control_name
        self.variants = variants
        self.results = {self.control_name: {"success": 0, "total": 0}}
        for v in variants:
            self.results[v] = {"success": 0, "total": 0}

    def record(self, variant, success):
        if variant in self.results:
            self.results[variant]["success"] += 1 if success else 0
            self.results[variant]["total"] += 1

    def conversion_rate(self, variant):
        r = self.results[variant]
        if r["total"] == 0:
            return 0.0
        return r["success"] / r["total"]

    def lift(self, variant):
        """Relative lift over control."""
        c = self.conversion_rate(self.control_name)
        v = self.conversion_rate(variant)
        if c == 0:
            return 0.0
        return (v - c) / c

    def z_score(self, variant):
        """Z-score for two-proportion z-test."""
        c = self.results[self.control_name]
        v = self.results[variant]
        if c["total"] == 0 or v["total"] == 0:
            return 0.0
        p_c = c["success"] / c["total"]
        p_v = v["success"] / v["total"]
        p_pool = (c["success"] + v["success"]) / (c["total"] + v["total"])
        if p_pool == 0 or p_pool == 1:
            return 0.0
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / c["total"] + 1 / v["total"]))
        if se == 0:
            return 0.0
        return (p_v - p_c) / se

    def is_significant(self, variant, threshold=1.96):
        return abs(self.z_score(variant)) >= threshold


def required_sample_size(baseline, mde, alpha=0.05, power=0.8):
    """Required sample size per variant for given baseline + min detectable effect."""
    z_alpha = 1.96
    z_beta = 0.84
    p1 = baseline
    p2 = baseline + mde
    if p1 == p2:
        return 0
    p_bar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / (p2 - p1) ** 2)


def main() -> int:
    test = ABTest("test", "A", ["B", "C"])
    test.record("A", True)
    test.record("A", False)
    test.record("B", True)
    test.record("B", True)
    test.record("B", False)
    test.record("C", False)
    test.record("C", False)
    print(f"Lift B: {test.lift('B'):.0%}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())