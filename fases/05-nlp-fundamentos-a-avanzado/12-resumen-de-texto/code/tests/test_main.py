"""Pruebas para 12-resumen-de-texto."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestRouge(unittest.TestCase):
    def test_rouge1_perfecto(self):
        ref = main.tokenizar("el gato come pescado")
        cand = main.tokenizar("el gato come pescado")
        self.assertAlmostEqual(main.rouge_n(ref, cand, 1), 1.0)

    def test_rouge1_parcial(self):
        ref = main.tokenizar("el gato come pescado")
        cand = main.tokenizar("el gato come")
        # 3/4 tokens en comun
        self.assertAlmostEqual(main.rouge_n(ref, cand, 1), 3 / 4)

    def test_rouge_l(self):
        ref = main.tokenizar("el gato come pescado")
        cand = main.tokenizar("el gato come")
        # LCS = 3, recall=3/4=0.75, precision=1.0, F1=6/7=0.857
        self.assertAlmostEqual(main.rouge_l(ref, cand), 6/7, places=3)

    def test_rouge_l_sin_match(self):
        ref = main.tokenizar("el gato come")
        cand = main.tokenizar("el perro ladra")
        # LCS = 1 ('el')
        # recall = 1/3, precision = 1/3
        # F1 = 1/3
        self.assertAlmostEqual(main.rouge_l(ref, cand), 1/3, places=2)


class TestExtractive(unittest.TestCase):
    def test_lead_n(self):
        texto = "Primera. Segunda. Tercera. Cuarta."
        out = main.extractive_lead_n(texto, n_oraciones=2)
        self.assertIn("Primera", out)
        self.assertIn("Segunda", out)
        self.assertNotIn("Tercera", out)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("ROUGE", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()