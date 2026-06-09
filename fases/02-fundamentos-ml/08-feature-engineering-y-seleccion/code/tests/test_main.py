"""Pruebas para 08-feature-engineering-y-seleccion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEstandarizar(unittest.TestCase):
    def test_media_cero(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        Xs = main.estandarizar(X)
        self.assertAlmostEqual(Xs.mean(), 0.0)

    def test_std_uno(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        Xs = main.estandarizar(X)
        self.assertAlmostEqual(Xs.std(), 1.0)


class TestMinMax(unittest.TestCase):
    def test_rango(self):
        X = np.array([[1.0, 10.0], [3.0, 30.0]])
        Xn = main.min_max(X)
        self.assertAlmostEqual(Xn.min(), 0.0)
        self.assertAlmostEqual(Xn.max(), 1.0)


class TestOneHot(unittest.TestCase):
    def test_forma(self):
        y = np.array([0, 1, 2])
        oh = main.one_hot(y)
        self.assertEqual(oh.shape, (3, 3))

    def test_unicidad_por_fila(self):
        y = np.array([0, 1, 2, 0])
        oh = main.one_hot(y)
        for fila in oh:
            self.assertEqual(fila.sum(), 1)


class TestPolynomial(unittest.TestCase):
    def test_grado_1(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Xp = main.polynomial_features(X, grado=1)
        self.assertEqual(Xp.shape, (2, 2))

    def test_grado_2(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Xp = main.polynomial_features(X, grado=2)
        # 2 (originales) + 1 (x1^2) + 1 (x2^2) + 1 (x1*x2) = 5
        self.assertEqual(Xp.shape, (2, 5))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("One-hot", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()