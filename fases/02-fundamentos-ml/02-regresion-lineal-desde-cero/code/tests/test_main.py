"""Pruebas para 02-regresion-lineal-desde-cero."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestAjuste(unittest.TestCase):
    def test_ajuste_perfecto(self):
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2 * X + 1
        w = main.ajustar(X, y)
        self.assertAlmostEqual(w[0], 1.0, places=6)
        self.assertAlmostEqual(w[1], 2.0, places=6)

    def test_ajuste_con_ruido(self):
        rng = np.random.default_rng(0)
        X = np.linspace(0, 10, 100)
        y = 3.0 * X + 0.5 + rng.normal(scale=0.1, size=100)
        w = main.ajustar(X, y)
        self.assertAlmostEqual(w[0], 0.5, places=1)
        self.assertAlmostEqual(w[1], 3.0, places=1)


class TestPrediccion(unittest.TestCase):
    def test_predice_sobre_datos_vistos(self):
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 5.0, 7.0])
        w = main.ajustar(X, y)
        y_pred = main.predecir(w, X)
        np.testing.assert_allclose(y_pred, y, atol=1e-6)


class TestMetricas(unittest.TestCase):
    def test_mse_perfecto(self):
        y = np.array([1, 2, 3])
        self.assertEqual(main.mse(y, y), 0.0)

    def test_r2_perfecto(self):
        y = np.array([1, 2, 3, 4, 5])
        self.assertEqual(main.r2(y, y), 1.0)

    def test_r2_baseline(self):
        y = np.array([1, 2, 3, 4, 5])
        # Si predecimos la media, R^2 = 0
        y_pred = np.full_like(y, y.mean())
        self.assertAlmostEqual(main.r2(y, y_pred), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("R^2", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()