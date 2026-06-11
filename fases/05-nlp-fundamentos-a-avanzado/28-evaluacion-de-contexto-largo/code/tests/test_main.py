"""Pruebas para 28-evaluacion-de-contexto-largo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestNeedle(unittest.TestCase):
    def test_encontrado(self):
        score = main.needle_haystack_test("Pedro tiene 42", "Pedro tiene 42", "...")
        self.assertEqual(score, 1.0)

    def test_no_encontrado(self):
        score = main.needle_haystack_test("Maria", "Pedro tiene 42", "...")
        self.assertEqual(score, 0.0)


class TestPosicion(unittest.TestCase):
    def test_posicion_inicio(self):
        pos = main.posicion_needle("hola", "hola mundo")
        # 'hola' esta en el inicio
        self.assertLess(pos, 0.1)

    def test_no_encontrado(self):
        self.assertIsNone(main.posicion_needle("xyz", "abc def"))


class TestScore(unittest.TestCase):
    def test_promedio(self):
        scores = {0.0: 1.0, 0.5: 0.5}
        self.assertEqual(main.score_por_posicion(scores), 0.75)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Encontrado", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()