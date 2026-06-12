"""Pruebas para 21-agent-economies."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestBudget(unittest.TestCase):
    def test_create(self):
        b = main.Budget(100)
        self.assertEqual(b.remaining, 100)

    def test_spend(self):
        b = main.Budget(100)
        ok, _ = b.spend(30)
        self.assertTrue(ok)
        self.assertEqual(b.remaining, 70)

    def test_spend_exceeds(self):
        b = main.Budget(50)
        ok, reason = b.spend(60)
        self.assertFalse(ok)
        self.assertEqual(reason, "exceeded")

    def test_reset(self):
        b = main.Budget(50)
        b.spend(40)
        b.reset()
        self.assertEqual(b.remaining, 50)

    def test_spent(self):
        b = main.Budget(50)
        b.spend(20)
        self.assertEqual(b.spent(), 20)


class TestMarket(unittest.TestCase):
    def test_list(self):
        m = main.ResourceMarket()
        m.list_resource("gpu", 10, reserve_price=1.0)
        self.assertIn("gpu", m.resources)

    def test_bid(self):
        m = main.ResourceMarket()
        m.list_resource("gpu", 1)
        m.submit_bid("a1", "gpu", 5.0)
        m.submit_bid("a2", "gpu", 3.0)
        winner = m.run_auction("gpu")
        self.assertEqual(winner["agent"], "a1")

    def test_no_bids(self):
        m = main.ResourceMarket()
        m.list_resource("gpu", 1)
        self.assertIsNone(m.run_auction("gpu"))


class TestAllocate(unittest.TestCase):
    def test_allocate(self):
        budgets = {
            "a1": {"obj": main.Budget(100), "remaining": 100},
            "a2": {"obj": main.Budget(50), "remaining": 50},
        }
        costs = {"task1": 30, "task2": 20}
        result = main.allocate_budget(budgets, costs)
        self.assertEqual(len(result), 2)


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