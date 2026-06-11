"""Pruebas para 10-evaluacion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestLMJudge(unittest.TestCase):
    def test_perfect(self):
        s = main.lm_judge_score("Q?", "AI is intelligence", "AI is intelligence")
        self.assertEqual(s, 1.0)

    def test_empty(self):
        self.assertEqual(main.lm_judge_score("Q?", "", "answer"), 0.0)


class TestABTest(unittest.TestCase):
    def test_basic(self):
        # 60 vs 60: very similar
        p = main.ab_test_se(60, 60)
        self.assertGreater(p, 0.05)


class TestPrecisionRecall(unittest.TestCase):
    def test_precision(self):
        p = main.precision_at_k(["d1", "d2"], ["d1", "d3", "d2", "d4"], k=2)
        # top-2: [d1, d3], relevant: d1 -> 1/2
        self.assertEqual(p, 0.5)

    def test_recall(self):
        r = main.recall_at_k(["d1", "d2", "d3"], ["d1", "d3", "d4"], k=2)
        # top-2: [d1, d3], relevant retrieved: d1, d3 -> 2/3
        self.assertAlmostEqual(r, 2 / 3, places=4)


class TestBLEU(unittest.TestCase):
    def test_perfect(self):
        s = main.bleu_score("a b c d", "a b c d")
        self.assertGreater(s, 0.9)

    def test_zero(self):
        s = main.bleu_score("a b c", "x y z")
        self.assertEqual(s, 0.0)


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