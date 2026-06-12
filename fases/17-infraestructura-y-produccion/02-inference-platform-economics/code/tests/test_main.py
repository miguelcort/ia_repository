"""Pruebas para 02-inference-platform-economics."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPlan(unittest.TestCase):
    def test_cost_basic(self):
        p = main.InferencePlan("x", 1e-8, 3e-8)
        c = p.cost(1_000_000, 1_000_000)
        self.assertAlmostEqual(c, 0.04, places=4)

    def test_batch_discount(self):
        p = main.InferencePlan("x", 1e-8, 3e-8, batch_discount=0.5)
        c = p.cost(1_000_000, 1_000_000, batch=True)
        self.assertAlmostEqual(c, 0.02, places=4)

    def test_reserved_discount(self):
        p = main.InferencePlan("x", 1e-8, 3e-8, reserved_discount=0.3)
        c = p.cost(1_000_000, 1_000_000, reserved=True)
        self.assertAlmostEqual(c, 0.028, places=4)

    def test_combined(self):
        p = main.InferencePlan("x", 1e-8, 3e-8, batch_discount=0.5, reserved_discount=0.3)
        c = p.cost(1_000_000, 1_000_000, batch=True, reserved=True)
        self.assertAlmostEqual(c, 0.014, places=4)


class TestTCO(unittest.TestCase):
    def test_basic(self):
        p = main.InferencePlan("x", 1e-8, 3e-8)
        tco = main.tco(p, 1_000_000, 1_000_000)
        self.assertAlmostEqual(tco, 0.04, places=4)

    def test_with_batch(self):
        p = main.InferencePlan("x", 1e-8, 3e-8, batch_discount=0.5)
        tco = main.tco(p, 1_000_000, 1_000_000, batch_ratio=0.5)
        self.assertAlmostEqual(tco, 0.03, places=4)

    def test_with_fixed(self):
        p = main.InferencePlan("x", 1e-8, 3e-8)
        tco = main.tco(p, 0, 0, fixed_monthly=100.0)
        self.assertEqual(tco, 100.0)


class TestPerRequest(unittest.TestCase):
    def test_basic(self):
        p = main.InferencePlan("x", 1e-5, 3e-5)
        c = main.per_request_cost(p, 1000, 500)
        self.assertAlmostEqual(c, 0.025, places=4)


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