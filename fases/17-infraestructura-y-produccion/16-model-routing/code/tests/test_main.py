"""Pruebas para 16-model-routing."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRoute(unittest.TestCase):
    def test_create(self):
        r = main.ModelRoute("x", 0.5, 5, 1.0)
        self.assertEqual(r.name, "x")


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.r = main.ModelRouter(strategy="cheapest")
        self.r.add_route(main.ModelRoute("haiku", 0.25, 3, 0.5))
        self.r.add_route(main.ModelRoute("sonnet", 3.0, 8, 2.0))
        self.r.add_route(main.ModelRoute("opus", 15.0, 10, 5.0))

    def test_route_cheapest(self):
        r = self.r.route(complexity=2)
        self.assertEqual(r.name, "haiku")

    def test_route_above_capability(self):
        r = self.r.route(complexity=9)
        self.assertEqual(r.name, "opus")

    def test_route_no_match(self):
        r = main.ModelRouter()
        self.assertIsNone(r.route(complexity=10))

    def test_route_fastest(self):
        r = main.ModelRouter(strategy="fastest")
        r.add_route(main.ModelRoute("slow", 1.0, 5, 10.0))
        r.add_route(main.ModelRoute("fast", 2.0, 5, 1.0))
        result = r.route(complexity=4)
        self.assertEqual(result.name, "fast")

    def test_route_best(self):
        r = main.ModelRouter(strategy="best")
        r.add_route(main.ModelRoute("low", 1.0, 3, 1.0))
        r.add_route(main.ModelRoute("high", 5.0, 8, 5.0))
        result = r.route(complexity=4)
        self.assertEqual(result.name, "high")

    def test_estimate_cost(self):
        r = self.r.route(complexity=2)
        cost = self.r.estimate_cost(r, 1000)
        self.assertAlmostEqual(cost, 0.25, places=4)


class TestComplexity(unittest.TestCase):
    def test_short(self):
        self.assertLess(main.detect_complexity("hi"), 1.0)

    def test_long(self):
        c = main.detect_complexity("a" * 1000)
        self.assertGreaterEqual(c, 5.0)

    def test_keywords(self):
        c = main.detect_complexity("analyze this complex code")
        self.assertGreater(c, 0)


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