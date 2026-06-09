"""Pruebas para 17-sistemas-lineales."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestResolver(unittest.TestCase):
    def test_sistema_2x2(self):
        A = np.array([[3.0, 2.0], [1.0, 2.0]])
        b = np.array([7.0, 5.0])
        x = main.resolver(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-10)

    def test_sistema_3x3(self):
        rng = np.random.default_rng(0)
        A = rng.random((3, 3)) + np.eye(3)
        b = rng.random(3)
        x = main.resolver(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-9)


class TestLSTSQ(unittest.TestCase):
    def test_ajuste_minimos_cuadrados(self):
        # y = 2x + 1 + ruido
        rng = np.random.default_rng(0)
        x_real = np.linspace(0, 10, 50)
        y = 2 * x_real + 1 + 0.1 * rng.normal(size=50)
        A = np.column_stack([x_real, np.ones_like(x_real)])
        coef = main.lstsq(A, y)
        self.assertAlmostEqual(coef[0], 2.0, places=1)
        self.assertAlmostEqual(coef[1], 1.0, places=1)


class TestCondicion(unittest.TestCase):
    def test_identidad(self):
        self.assertAlmostEqual(main.condicion(np.eye(3)), 1.0)

    def test_casi_singular(self):
        A = np.array([[1.0, 1.0], [1.0, 1.0001]])
        self.assertGreater(main.condicion(A), 1000)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("x =", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()