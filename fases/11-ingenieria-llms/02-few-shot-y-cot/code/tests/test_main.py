"""Pruebas para 02-few-shot-y-cot."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestFewShot(unittest.TestCase):
    def test_basic(self):
        examples = [("2+2", "4"), ("3+3", "6")]
        p = main.few_shot_prompt("5+5", examples)
        self.assertIn("2+2", p)
        self.assertIn("5+5", p)
        self.assertIn("4", p)


class TestCoT(unittest.TestCase):
    def test_cot(self):
        p = main.cot_prompt("Cual es 5+5?")
        self.assertIn("paso a paso", p)


class TestSelfConsistency(unittest.TestCase):
    def test_majority(self):
        ans = main.self_consistency_answer(["A", "A", "B", "A", "C"])
        self.assertEqual(ans, "A")

    def test_tie(self):
        # 2 A, 2 B -> first encountered
        ans = main.self_consistency_answer(["A", "B", "A", "B"])
        self.assertIn(ans, ["A", "B"])


class TestDiversity(unittest.TestCase):
    def test_high(self):
        s = main.diversity_score(["A", "B", "C", "D"])
        self.assertEqual(s, 1.0)

    def test_zero(self):
        s = main.diversity_score(["A", "A", "A", "B"])
        # 2/4 = 0.5
        self.assertAlmostEqual(s, 0.5, places=4)


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