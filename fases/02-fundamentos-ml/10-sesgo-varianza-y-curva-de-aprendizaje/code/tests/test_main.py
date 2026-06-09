"""Pruebas para 10-sesgo-varianza-y-curva-de-aprendizaje."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMSE(unittest.TestCase):
    def test_perfecto(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(main.mse(y, y), 0.0)

    def test_diferencia(self):
        y = np.array([1.0, 2.0, 3.0])
        y_p = np.array([2.0, 3.0, 4.0])
        self.assertAlmostEqual(main.mse(y, y_p), 1.0)


class TestCurvaAprendizaje(unittest.TestCase):
    def test_longitud(self):
        rng = np.random.default_rng(0)
        X = np.linspace(0, 10, 30).reshape(-1, 1)
        y = 2 * X.squeeze() + 1 + 0.1 * rng.normal(size=30)
        sizes, tr, va = main.curva_aprendizaje(lambda Xt, yt, Xe: 2 * Xe.squeeze() + 1, X, y, k=5)
        self.assertEqual(len(sizes), len(tr))
        self.assertEqual(len(sizes), len(va))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Tamanos", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()