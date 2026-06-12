"""Pruebas para 14-kill-switches-canaries."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestFeatureFlag(unittest.TestCase):
    def test_disabled_by_default(self):
        f = main.FeatureFlag("x")
        self.assertFalse(f.is_enabled_for("u1"))

    def test_enabled_full(self):
        f = main.FeatureFlag("x", enabled=True, rollout_pct=100)
        for i in range(10):
            self.assertTrue(f.is_enabled_for(f"u{i}"))

    def test_rollout_zero(self):
        f = main.FeatureFlag("x", enabled=True, rollout_pct=0)
        self.assertFalse(f.is_enabled_for("u1"))

    def test_allowlist(self):
        f = main.FeatureFlag("x", enabled=False, allowlist={"vip"})
        self.assertTrue(f.is_enabled_for("vip"))
        self.assertFalse(f.is_enabled_for("normal"))

    def test_kill_switch(self):
        f = main.FeatureFlag("x", enabled=True, rollout_pct=100)
        f.kill()
        self.assertFalse(f.is_enabled_for("u1"))
        self.assertEqual(f.rollout_pct, 0.0)

    def test_enable(self):
        f = main.FeatureFlag("x", enabled=False)
        f.enable()
        self.assertTrue(f.enabled)

    def test_set_rollout_invalid(self):
        f = main.FeatureFlag("x")
        with self.assertRaises(ValueError):
            f.set_rollout(150)

    def test_rollout_partial(self):
        f = main.FeatureFlag("x", enabled=True, rollout_pct=50.0)
        hits = sum(1 for i in range(1000) if f.is_enabled_for(f"u{i}"))
        self.assertGreater(hits, 350)
        self.assertLess(hits, 650)


class TestCanaryDeploy(unittest.TestCase):
    def test_route_distribution(self):
        c = main.CanaryDeploy(canary_pct=10)
        counts = {"canary": 0, "baseline": 0}
        for i in range(2000):
            counts[c.route(f"req-{i}")] += 1
        self.assertGreater(counts["canary"], 100)
        self.assertLess(counts["canary"], 300)

    def test_promote_when_healthy(self):
        c = main.CanaryDeploy(canary_pct=100, min_requests=10, error_threshold=0.1)
        for i in range(20):
            c.record(f"r-{i}", "canary", True)
        self.assertTrue(c.should_promote())

    def test_no_promote_when_below_min(self):
        c = main.CanaryDeploy(canary_pct=100, min_requests=100)
        self.assertFalse(c.should_promote())

    def test_rollback_on_high_error(self):
        c = main.CanaryDeploy(canary_pct=100, min_requests=10, error_threshold=0.1)
        for i in range(20):
            c.record(f"r-{i}", "canary", i < 5)
        self.assertTrue(c.should_rollback())

    def test_no_rollback_when_healthy(self):
        c = main.CanaryDeploy(canary_pct=100, min_requests=10, error_threshold=0.1)
        for i in range(20):
            c.record(f"r-{i}", "canary", True)
        self.assertFalse(c.should_rollback())


class TestHealthCheck(unittest.TestCase):
    def test_healthy(self):
        h = main.HealthCheck("api", lambda: True, unhealthy_threshold=3)
        h.run()
        self.assertTrue(h.is_healthy)

    def test_unhealthy_after_threshold(self):
        h = main.HealthCheck("api", lambda: False, unhealthy_threshold=2)
        h.run()
        h.run()
        self.assertFalse(h.is_healthy)

    def test_recovers(self):
        n = [0]
        def fn():
            n[0] += 1
            return n[0] > 2
        h = main.HealthCheck("api", fn, unhealthy_threshold=2)
        h.run()
        h.run()
        self.assertFalse(h.is_healthy)
        h.run()
        self.assertTrue(h.is_healthy)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()