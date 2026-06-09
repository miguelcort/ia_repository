"""Pruebas para 05-support-vector-machines."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestKernels(unittest.TestCase):
    def test_kernel_lineal(self):
        self.assertEqual(main.kernel_lineal([1, 0], [0, 1]), 0.0)
        # dot([1,1], [2,2]) = 1*2 + 1*2 = 4
        self.assertEqual(main.kernel_lineal([1, 1], [2, 2]), 4.0)
        # dot([1,2,3], [4,5,6]) = 4+10+18 = 32
        self.assertEqual(main.kernel_lineal([1, 2, 3], [4, 5, 6]), 32.0)

    def test_kernel_rbf_mismo(self):
        # K(x, x) = exp(0) = 1
        self.assertAlmostEqual(main.kernel_rbf([1, 2], [1, 2]), 1.0)

    def test_kernel_rbf_distinto(self):
        # K([0], [10], gamma=1) = exp(-1 * 100) ~ 0
        self.assertLess(main.kernel_rbf([0], [10], gamma=1.0), 1e-40)

    def test_kernel_rbf_gamma(self):
        # Con gamma=0.5, K(0, 2) = exp(-0.5 * 4) = exp(-2) ~ 0.135
        import math
        self.assertAlmostEqual(main.kernel_rbf([0], [2], gamma=0.5), math.exp(-2))


class TestMargen(unittest.TestCase):
    def test_margen_positivo(self):
        # X con intercepto en columna 0
        X = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        y = np.array([1, -1])
        w = np.array([0.0, 1.0, 0.0])  # hiperplano x1=0
        # margen = min(y * (X @ w)) / ||w||
        # = min(1*1.0, -1*-1.0) / 1 = 1.0
        m = main.margen(X, y, w)
        self.assertAlmostEqual(m, 1.0)

    def test_margen_con_intercepto(self):
        X = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        y = np.array([1, 1])
        w = np.array([1.0, 0.0, 0.0])  # hiperplano x1=0 con b=1
        # scores = [1, 1]; min(1*1, 1*1)/||[1,0,0]|| = 1
        m = main.margen(X, y, w)
        self.assertAlmostEqual(m, 1.0)


class TestHingeLoss(unittest.TestCase):
    def test_perfecto(self):
        y = np.array([1, -1])
        scores = np.array([10.0, -10.0])
        # Sin regularizacion, hinge = 0
        self.assertAlmostEqual(main.hinge_loss(y, scores, reg=0.0), 0.0)

    def test_mal(self):
        y = np.array([1, 1])
        scores = np.array([-1.0, -1.0])
        # Sin regularizacion, max(0, 1-y*s) = 2 para cada uno
        self.assertAlmostEqual(main.hinge_loss(y, scores, reg=0.0), 2.0)

    def test_con_regularizacion(self):
        y = np.array([1, -1])
        scores = np.array([2.0, -2.0])
        # max(0, 1-1*2)=0, max(0, 1-(-1)*(-2))=0, reg=1, ||s||^2=8
        esperado = 0.0 + 0.5 * 1 * 8
        self.assertAlmostEqual(main.hinge_loss(y, scores), esperado)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Margen", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()