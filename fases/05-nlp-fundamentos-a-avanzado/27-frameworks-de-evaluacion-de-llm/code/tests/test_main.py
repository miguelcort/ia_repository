"""Pruebas para 27-frameworks-de-evaluacion-de-llm."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEM(unittest.TestCase):
    def test_em(self):
        self.assertEqual(main.exact_match("Maria", "Maria"), 1.0)
        self.assertEqual(main.exact_match("Maria", "Pedro"), 0.0)


class TestF1(unittest.TestCase):
    def test_perfecto(self):
        self.assertEqual(main.f1_token("a b c", "a b c"), 1.0)

    def test_parcial(self):
        self.assertAlmostEqual(main.f1_token("a", "a b"), 2/3, places=2)


class TestRougeL(unittest.TestCase):
    def test_perfecto(self):
        self.assertEqual(main.rouge_l("a b c", "a b c"), 1.0)


class TestFaithfulness(unittest.TestCase):
    def test_alta(self):
        # Si todos los tokens de la respuesta estan en el contexto
        s = main.faithfulness_score("Maria Madrid", "Maria vive en Madrid")
        self.assertGreater(s, 0.5)


class TestLLMJudge(unittest.TestCase):
    def test_score_rango(self):
        score = main.llm_as_judge_mock("pregunta", "Maria", "Maria")
        self.assertGreaterEqual(score, 1)
        self.assertLessEqual(score, 5)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("EM", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()