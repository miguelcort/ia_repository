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


class TestBenchmarks(unittest.TestCase):
    def test_diez(self):
        b = main.benchmarks_summary()
        self.assertEqual(len(b), 10)
        # Comprobar keys contienen los nombres
        all_keys = " ".join(b.keys())
        self.assertIn("MMLU", all_keys)
        self.assertIn("HumanEval", all_keys)


class TestLMJudge(unittest.TestCase):
    def test_perfect(self):
        score = main.lm_as_judge_score("hola mundo", "hola mundo", "")
        self.assertEqual(score, 1.0)

    def test_no_match(self):
        score = main.lm_as_judge_score("foo bar", "hello world", "")
        self.assertEqual(score, 0.0)

    def test_empty_ref(self):
        score = main.lm_as_judge_score("hola", "", "")
        self.assertEqual(score, 0.5)


class TestPassAtK(unittest.TestCase):
    def test_perfect(self):
        p = main.pass_at_k(n_samples=10, n_correct=10, k=1)
        self.assertEqual(p, 1.0)

    def test_zero(self):
        p = main.pass_at_k(n_samples=10, n_correct=0, k=1)
        self.assertEqual(p, 0.0)

    def test_k_works(self):
        # 5 correct de 10, k=2
        p = main.pass_at_k(10, 5, 2)
        self.assertGreater(p, 0.5)


class TestElo(unittest.TestCase):
    def test_a_gana(self):
        a, b = main.elo_rating(1500, 1500, score_a=1.0)
        self.assertGreater(a, 1500)


class TestFrameworks(unittest.TestCase):
    def test_seis(self):
        f = main.evals_framework()
        self.assertEqual(len(f), 7)


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