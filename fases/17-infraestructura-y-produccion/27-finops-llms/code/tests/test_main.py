"""Pruebas para 27-finops-llms."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRecord(unittest.TestCase):
    def test_create(self):
        r = main.CostRecord("team_a", "openai", 100.0, 1000, "2024-Q1")
        self.assertEqual(r.team, "team_a")
        self.assertEqual(r.cost_usd, 100.0)


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.t = main.FinOpsTracker()
        self.t.set_budget("a", 1000.0)
        self.t.set_budget("b", 500.0)
        self.t.record(main.CostRecord("a", "openai", 250.0, 1_000_000, "Q1"))
        self.t.record(main.CostRecord("a", "anthropic", 100.0, 500_000, "Q1"))
        self.t.record(main.CostRecord("b", "openai", 200.0, 800_000, "Q1"))

    def test_total_cost(self):
        self.assertEqual(self.t.total_cost(), 550.0)

    def test_total_cost_by_team(self):
        self.assertEqual(self.t.total_cost(team="a"), 350.0)

    def test_total_cost_by_period(self):
        self.assertEqual(self.t.total_cost(period="Q1"), 550.0)

    def test_cost_by_team(self):
        cb = self.t.cost_by_team()
        self.assertEqual(cb["a"], 350.0)
        self.assertEqual(cb["b"], 200.0)

    def test_budget_utilization(self):
        self.assertEqual(self.t.budget_utilization("a"), 0.35)

    def test_budget_no_set(self):
        self.assertEqual(self.t.budget_utilization("c"), 0.0)

    def test_unit_economics(self):
        cost_per_token = self.t.unit_economics("a", "Q1")
        self.assertAlmostEqual(cost_per_token, 350 / 1_500_000, places=9)

    def test_unit_economics_empty(self):
        self.assertEqual(self.t.unit_economics("c", "Q1"), 0.0)


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