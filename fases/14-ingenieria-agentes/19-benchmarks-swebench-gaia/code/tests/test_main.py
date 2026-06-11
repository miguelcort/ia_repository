"""Pruebas para 19-benchmarks-swebench-gaia."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSWEbench(unittest.TestCase):
    def test_basic(self):
        preds = ["a", "b", "a", "c"]
        truth = ["a", "b", "c", "c"]
        scores = main.swebench_score(preds, truth)
        # 3 correct out of 4 -> 0.75
        self.assertEqual(scores["pass@1"], 0.75)

    def test_all_correct(self):
        preds = ["a", "b"]
        truth = ["a", "b"]
        scores = main.swebench_score(preds, truth)
        self.assertEqual(scores["pass@1"], 1.0)

    def test_none_correct(self):
        preds = ["a", "b"]
        truth = ["c", "d"]
        scores = main.swebench_score(preds, truth)
        self.assertEqual(scores["pass@1"], 0.0)

    def test_empty(self):
        scores = main.swebench_score([], [])
        self.assertEqual(scores, {})


class TestGAIA(unittest.TestCase):
    def test_basic(self):
        truth = [
            {"level": "level1", "answer": "a"},
            {"level": "level1", "answer": "b"},
            {"level": "level2", "answer": "c"},
        ]
        preds = ["a", "b", "x"]  # level1: 2/2, level2: 0/1
        scores = main.gaia_score(preds, truth)
        self.assertEqual(scores["level1"], 1.0)
        self.assertEqual(scores["level2"], 0.0)


class TestSuccessRate(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(main.success_rate(["a", "b"], ["a", "b"]), 1.0)
        self.assertEqual(main.success_rate(["a"], ["b"]), 0.0)
        self.assertEqual(main.success_rate([], []), 0.0)


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