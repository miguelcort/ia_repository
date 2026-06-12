"""
Lección: 14-kill-switches-canaries
Fase: 15
Kill switches + canaries: feature flags,
gradual rollout, automatic rollback, circuit
breakers, health checks, canary analysis.
"""
from __future__ import annotations
import time
import uuid


class FeatureFlag:
    def __init__(self, name, enabled=False, rollout_pct=0.0, allowlist=None):
        self.name = name
        self.enabled = enabled
        self.rollout_pct = rollout_pct
        self.allowlist = allowlist or set()
        self.denylist = set()

    def is_enabled_for(self, user_id):
        if self.name in self.denylist or user_id in self.denylist:
            return False
        if user_id in self.allowlist:
            return True
        if not self.enabled:
            return False
        if self.rollout_pct >= 100:
            return True
        bucket = (hash(user_id) % 10000) / 100.0
        return bucket < self.rollout_pct

    def set_rollout(self, pct):
        if not 0 <= pct <= 100:
            raise ValueError("pct must be 0-100")
        self.rollout_pct = pct

    def kill(self):
        self.enabled = False
        self.rollout_pct = 0.0

    def enable(self):
        self.enabled = True


class CanaryDeploy:
    def __init__(self, canary_pct=10, error_threshold=0.05,
                 window_seconds=300, min_requests=100):
        self.canary_pct = canary_pct
        self.error_threshold = error_threshold
        self.window_seconds = window_seconds
        self.min_requests = min_requests
        self.canary_requests = []
        self.baseline_requests = []

    def route(self, request_id):
        bucket = (hash(request_id) % 10000) / 100.0
        return "canary" if bucket < self.canary_pct else "baseline"

    def record(self, request_id, variant, success):
        entry = (time.time(), success)
        if variant == "canary":
            self.canary_requests.append(entry)
        else:
            self.baseline_requests.append(entry)
        self._gc()

    def _gc(self):
        cutoff = time.time() - self.window_seconds
        self.canary_requests = [e for e in self.canary_requests if e[0] >= cutoff]
        self.baseline_requests = [e for e in self.baseline_requests if e[0] >= cutoff]

    def should_promote(self):
        if len(self.canary_requests) < self.min_requests:
            return False
        c_errors = sum(1 for _, s in self.canary_requests if not s)
        c_err_rate = c_errors / len(self.canary_requests)
        if c_err_rate > self.error_threshold:
            return False
        if len(self.baseline_requests) >= self.min_requests:
            b_errors = sum(1 for _, s in self.baseline_requests if not s)
            b_err_rate = b_errors / len(self.baseline_requests)
            return c_err_rate <= b_err_rate
        return True

    def should_rollback(self):
        if len(self.canary_requests) < self.min_requests:
            return False
        c_errors = sum(1 for _, s in self.canary_requests if not s)
        c_err_rate = c_errors / len(self.canary_requests)
        return c_err_rate > self.error_threshold


class HealthCheck:
    def __init__(self, name, check_fn, interval_seconds=30, unhealthy_threshold=3):
        self.name = name
        self.check_fn = check_fn
        self.interval_seconds = interval_seconds
        self.unhealthy_threshold = unhealthy_threshold
        self.consecutive_failures = 0
        self.last_check = 0
        self.last_status = None

    def run(self):
        self.last_check = time.time()
        try:
            self.last_status = self.check_fn()
            if self.last_status:
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1
        except Exception:
            self.consecutive_failures += 1
        return self.is_healthy

    @property
    def is_healthy(self):
        return self.consecutive_failures < self.unhealthy_threshold


def main() -> int:
    flag = FeatureFlag("new_ui", enabled=True, rollout_pct=50.0)
    print(flag.is_enabled_for("user-1"))
    print(flag.is_enabled_for("user-2"))
    flag.kill()
    print(flag.is_enabled_for("user-1"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())