"""Pruebas para 08-inicializacion-de-pesos."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestInicializaciones(unittest.TestCase):
    def test_ceros(self):
        W = main.ceros(3, 4)
        np.testing.assert_array_equal(W, np.zeros((3, 4)))

    def test_aleatorio_simple_shape(self):
        W = main.aleatorio_simple(3, 4)
        self.assertEqual(W.shape, (3, 4))

    def test_xavier_varianza(self):
        # Para Xavier, varianza deberia ser ~ 1/n_in
        W = main.xavier_glorot(1000, 1000, semilla=0)
        # La varianza empirica deberia ser cercana a 1/1000 = 0.001
        self.assertAlmostEqual(W.var(), 0.001, places=4)

    def test_he_varianza(self):
        # Para He, varianza deberia ser ~ 2/n_in
        W = main.he(1000, 1000, semilla=0)
        # ~ 2/1000 = 0.002
        self.assertAlmostEqual(W.var(), 0.002, places=4)

    def test_lecun_varianza(self):
        W = main.lecun(1000, 1000, semilla=0)
        self.assertAlmostEqual(W.var(), 0.001, places=4)

    def test_reproducibilidad(self):
        W1 = main.he(10, 10, semilla=42)
        W2 = main.he(10, 10, semilla=42)
        np.testing.assert_array_equal(W1, W2)


class TestEstabilidad(unittest.TestCase):
    def test_he_estable(self):
        # He init debe mantener varianza estable a traves de 10 capas
        W = main.he(100, 100, semilla=0)
        vars_ = main.varianza_activaciones(100, W, n_pasos=10)
        # Varianza no debe explotar (>10) ni colapsar (<0.01)
        self.assertLess(max(vars_), 10.0)
        self.assertGreater(min(vars_), 0.01)

    def test_ceros_colapsa(self):
        # Ceros: varianza se mantiene 0
        W = main.ceros(100, 100)
        vars_ = main.varianza_activaciones(100, W, n_pasos=10)
        for v in vars_:
            self.assertEqual(v, 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("He", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()