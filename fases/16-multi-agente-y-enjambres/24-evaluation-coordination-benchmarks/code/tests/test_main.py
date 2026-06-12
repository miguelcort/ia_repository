"""Pruebas para 24-evaluation-coordination-benchmarks."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestBenchmarks(unittest.TestCase):
    def test_list(self):
        b = main.list_benchmarks()
        self.assertIn("MGSM", b)
        self.assertIn("HumanEval-Multi", b)

    def test_get(self):
        m = main.get_benchmark("MGSM")
        self.assertIn("metric", m)


class TestNormalize(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(main.normalize_score(80, "MGSM"), 0.8)

    def test_zero(self):
        self.assertEqual(main.normalize_score(0, "MGSM"), 0.0)

    def test_full(self):
        self.assertEqual(main.normalize_score(100, "MGSM"), 1.0)

    def test_overflow(self):
        self.assertEqual(main.normalize_score(150, "MGSM"), 1.0)

    def test_elo(self):
        result = main.normalize_score(1200, "ChatBot-Arena-MT")
        self.assertAlmostEqual(result, 0.5, places=2)

    def test_unknown(self):
        self.assertIsNone(main.normalize_score(50, "unknown"))


class TestAverage(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(main.average_coordination_score({"MGSM": 0.8, "HM": 0.6}), 0.7)

    def test_empty(self):
        self.assertEqual(main.average_coordination_score({}), 0.0)

    def test_with_none(self):
        self.assertEqual(main.average_coordination_score({"a": 0.5, "b": None}), 0.5)


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