"""Pruebas para 11-traduccion-automatica."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestBLEU(unittest.TestCase):
    def test_perfecto(self):
        ref = main.tokenizar("el gato come pescado")
        cand = main.tokenizar("el gato come pescado")
        self.assertAlmostEqual(main.bleu_score(ref, cand), 1.0, places=3)

    def test_parcial(self):
        ref = main.tokenizar("el gato come pescado")
        cand = main.tokenizar("el gato come")
        # BP < 1 porque cand es mas corto
        # 1-grama: 3/3 = 1, 2-grama: 2/2 = 1, 3-grama: 1/1 = 1
        # Pero BP = exp(1 - 4/3) = exp(-0.333) = 0.717
        score = main.bleu_score(ref, cand)
        # BLEU deberia ser ~0.72 (BP), < 1
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.0)

    def test_cero_match(self):
        ref = main.tokenizar("el gato come pescado")
        cand = main.tokenizar("el perro ladra fuerte")
        # Ningun n-grama en comun a partir de bigrama
        # 1-grama: 1/4 = 0.25 (solo 'el')
        score = main.bleu_score(ref, cand)
        self.assertLess(score, 0.3)

    def test_vacio(self):
        self.assertEqual(main.bleu_score([], []), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("BLEU", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()