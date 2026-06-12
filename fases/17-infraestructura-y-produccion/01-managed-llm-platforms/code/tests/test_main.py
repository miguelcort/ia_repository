"""Pruebas para 01-managed-llm-platforms."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPlatforms(unittest.TestCase):
    def test_list(self):
        ps = main.list_platforms()
        self.assertIn("openai", ps)
        self.assertIn("anthropic", ps)
        self.assertIn("google", ps)
        self.assertIn("bedrock", ps)
        self.assertIn("azure", ps)
        self.assertIn("together", ps)
        self.assertIn("fireworks", ps)

    def test_get(self):
        p = main.get_platform("openai")
        self.assertIn("models", p)
        self.assertIn("pricing_per_1m_input", p)
        self.assertIn("context_window", p)

    def test_get_unknown(self):
        self.assertIsNone(main.get_platform("unknown"))


class TestCompare(unittest.TestCase):
    def test_cheapest(self):
        platform_key, info = main.cheapest()
        self.assertIn(platform_key, main.MANAGED_PLATFORMS)

    def test_largest_context(self):
        platform_key, info = main.largest_context()
        self.assertIn(platform_key, main.MANAGED_PLATFORMS)
        self.assertGreaterEqual(info["context_window"], 128000)


class TestEstimate(unittest.TestCase):
    def test_basic(self):
        cost = main.estimate_cost("openai", 1_000_000, 100_000)
        self.assertAlmostEqual(cost, 2.5 + 1.0, places=2)

    def test_zero(self):
        self.assertEqual(main.estimate_cost("openai", 0, 0), 0.0)

    def test_unknown(self):
        self.assertIsNone(main.estimate_cost("unknown", 100, 100))


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