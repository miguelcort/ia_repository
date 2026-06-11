"""Pruebas para 13-preguntas-y-respuestas."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestQA(unittest.TestCase):
    def test_basico(self):
        qa = main.SimpleQASystem()
        qa.fit(["Maria vive en Madrid. Pedro trabaja en Barcelona."])
        res = qa.predecir("quien vive en Madrid", top_k=1)
        self.assertEqual(len(res), 1)
        self.assertIn("Maria", res[0])


class TestMetricas(unittest.TestCase):
    def test_em(self):
        self.assertEqual(main.exact_match("Maria", "Maria"), 1.0)
        self.assertEqual(main.exact_match("Maria", "maria"), 1.0)
        self.assertEqual(main.exact_match("Maria", "Pedro"), 0.0)

    def test_f1(self):
        # "Maria" vs "Maria": F1 = 1
        self.assertEqual(main.f1_score("Maria", "Maria"), 1.0)
        self.assertEqual(main.f1_score("Maria", "Pedro"), 0.0)
        # Parcial: "Maria" vs "Maria Madrid" -> P=1, R=1/2, F1=2/3
        self.assertAlmostEqual(main.f1_score("Maria", "Maria Madrid"), 2/3, places=2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Respuesta", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()