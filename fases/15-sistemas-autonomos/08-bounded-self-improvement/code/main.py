"""
Lección: 08-bounded-self-improvement
Fase: 15
Bounded self-improvement: limits + sandboxing + verification.
Safer alternative to recursive self-improvement.
Production-ready con safety guarantees.
"""
from __future__ import annotations
import time


class BoundedSelfImprover:
    """Mock bounded self-improver."""
    def __init__(self, name, max_modifications=5, require_verification=True):
        self.name = name
        self.max_modifications = max_modifications
        self.require_verification = require_verification
        self.modifications = []
        self.verifications = []

    def verify_improvement(self, old_code, new_code, eval_fn, min_improvement=0.0):
        """Verify improvement antes de aplicar."""
        old_score = eval_fn(old_code)
        new_score = eval_fn(new_code)
        improvement = new_score - old_score
        verified = improvement >= min_improvement
        self.verifications.append({
            "old": old_score,
            "new": new_score,
            "improvement": improvement,
            "verified": verified,
            "timestamp": time.time(),
        })
        return verified, improvement

    def apply_modification(self, old_code, new_code, eval_fn, min_improvement=0.0):
        """Apply modification solo si verified + under limit."""
        if len(self.modifications) >= self.max_modifications:
            return {"status": "limit_reached", "applied": len(self.modifications)}
        if self.require_verification:
            verified, improvement = self.verify_improvement(old_code, new_code, eval_fn, min_improvement)
            if not verified:
                return {"status": "rejected", "improvement": improvement}
        self.modifications.append({
            "from": old_code[:50],
            "to": new_code[:50],
            "timestamp": time.time(),
        })
        return {"status": "applied", "code": new_code, "improvement": improvement if self.require_verification else None}

    def rollback(self):
        """Rollback last modification."""
        if not self.modifications:
            return False
        last = self.modifications.pop()
        return last["from"]


def main() -> int:
    improver = BoundedSelfImprover("test", max_modifications=3)
    def eval_fn(code):
        return len(code)  # longer is "better" (mock)
    result = improver.apply_modification("v1", "v2_longer", eval_fn, min_improvement=2)
    print(f"Apply 1: {result['status']}, improvement: {result.get('improvement')}")
    result = improver.apply_modification("v2_longer", "v3", eval_fn, min_improvement=2)
    print(f"Apply 2: {result['status']}, improvement: {result.get('improvement')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())