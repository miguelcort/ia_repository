"""Pruebas para 18-optimizacion-convexa."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestConvexidad(unittest.TestCase):
    def test_cuadratica_estrictamente_convexa(self):
        f = lambda x, y: (x - 1) ** 2 + (y + 1) ** 2
        conv, _ = main.es_convexa_2d(f, [0.0, 0.0])
        self.assertTrue(conv)

    def test_silla_no_convexa(self):
        f = lambda x, y: x * y
        conv, autovals = main.es_convexa_2d(f, [0.0, 0.0])
        # La silla tiene un autovalor positivo y otro negativo
        self.assertFalse(conv)


class TestGradienteDescenso(unittest.TestCase):
    def test_encuentra_minimo_global_convexo(self):
        f = lambda x, y: (x - 3) ** 2 + (y + 2) ** 2
        w = main.gradiente_descenso_convexo(f, [0.0, 0.0], lr=0.1, max_iter=300)
        self.assertAlmostEqual(w[0], 3.0, places=1)
        self.assertAlmostEqual(w[1], -2.0, places=1)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Convexa", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()