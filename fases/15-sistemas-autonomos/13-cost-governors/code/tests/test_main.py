"""Pruebas para 13-cost-governors."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestCostGovernor(unittest.TestCase):
    def setUp(self):
        self.gov = main.CostGovernor(daily_budget_usd=1.0, per_request_cap=0.1)

    def test_estimate(self):
        c = self.gov.estimate_cost("gpt-4o", 1_000_000, 0)
        self.assertAlmostEqual(c, 5.0, places=4)

    def test_allow_within_budget(self):
        ok, cost, reason = self.gov.allow("gpt-4o-mini", 1000, 200)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_allow_exceeds_per_request(self):
        ok, cost, reason = self.gov.allow("gpt-4o", 1_000_000, 1_000_000)
        self.assertFalse(ok)
        self.assertEqual(reason, "exceeds_per_request_cap")

    def test_allow_exceeds_daily(self):
        big = main.CostGovernor(daily_budget_usd=0.1, per_request_cap=0.1)
        for _ in range(20):
            big.allow("gpt-4o", 10000, 5000)
        ok, cost, reason = big.allow("gpt-4o", 10000, 5000)
        self.assertFalse(ok)
        self.assertEqual(reason, "exceeds_daily_budget")

    def test_spent(self):
        self.gov.allow("gpt-4o-mini", 1000, 1000)
        self.assertGreater(self.gov.spent(), 0)

    def test_remaining(self):
        r1 = self.gov.remaining()
        self.gov.allow("gpt-4o-mini", 1000, 1000)
        r2 = self.gov.remaining()
        self.assertLess(r2, r1)

    def test_reset(self):
        self.gov.allow("gpt-4o-mini", 1000, 1000)
        self.gov.reset()
        self.assertEqual(self.gov.spent(), 0)

    def test_unknown_model_default(self):
        c = self.gov.estimate_cost("unknown-model", 1_000_000, 0)
        self.assertEqual(c, 5.0)


class TestPickCheapest(unittest.TestCase):
    def test_cheapest_picked(self):
        gov = main.CostGovernor()
        m, c = main.pick_cheapest_model(1000, gov)
        self.assertEqual(m, "gpt-4o-mini")
        self.assertGreater(c, 0)

    def test_candidates_filter(self):
        gov = main.CostGovernor()
        m, c = main.pick_cheapest_model(1000, gov, candidates=["claude-3-5-sonnet", "claude-3-haiku"])
        self.assertEqual(m, "claude-3-haiku")


class TestCircuitBreaker(unittest.TestCase):
    def test_closed_allows_call(self):
        cb = main.CircuitBreaker(failure_threshold=2, cooldown_seconds=10)
        result = cb.call(lambda: 42)
        self.assertEqual(result, 42)

    def test_opens_after_threshold(self):
        cb = main.CircuitBreaker(failure_threshold=2, cooldown_seconds=10)
        def boom():
            raise ValueError("fail")
        with self.assertRaises(ValueError):
            cb.call(boom)
        with self.assertRaises(ValueError):
            cb.call(boom)
        self.assertTrue(cb.is_open)

    def test_open_raises(self):
        cb = main.CircuitBreaker(failure_threshold=1, cooldown_seconds=10)
        def boom():
            raise ValueError("fail")
        with self.assertRaises(ValueError):
            cb.call(boom)
        with self.assertRaises(RuntimeError):
            cb.call(lambda: 1)


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