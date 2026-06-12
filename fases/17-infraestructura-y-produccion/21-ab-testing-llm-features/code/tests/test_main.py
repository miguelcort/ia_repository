"""Pruebas para 21-ab-testing-llm-features."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestAB(unittest.TestCase):
    def setUp(self):
        self.t = main.ABTest("test", "A", ["B", "C"])
        self.t.record("A", True)
        self.t.record("A", False)
        self.t.record("B", True)
        self.t.record("B", True)
        self.t.record("B", False)
        self.t.record("C", False)
        self.t.record("C", False)

    def test_conversion(self):
        self.assertAlmostEqual(self.t.conversion_rate("A"), 0.5, places=4)
        self.assertAlmostEqual(self.t.conversion_rate("B"), 2 / 3, places=4)
        self.assertEqual(self.t.conversion_rate("C"), 0.0)

    def test_lift(self):
        lift = self.t.lift("B")
        self.assertGreater(lift, 0)

    def test_z_score(self):
        z = self.t.z_score("B")
        self.assertIsNotNone(z)

    def test_significant(self):
        self.assertFalse(self.t.is_significant("C"))


class TestSampleSize(unittest.TestCase):
    def test_basic(self):
        n = main.required_sample_size(0.1, 0.02)
        self.assertGreater(n, 0)

    def test_zero_mde(self):
        self.assertEqual(main.required_sample_size(0.1, 0), 0)

    def test_larger_mde(self):
        small = main.required_sample_size(0.1, 0.02)
        large = main.required_sample_size(0.1, 0.05)
        self.assertLess(large, small)


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