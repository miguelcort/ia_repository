"""Pruebas para 18-seleccion-de-features."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVarianza(unittest.TestCase):
    def test_detecta_constante(self):
        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        mascara, var = main.varianza_threshold(X, umbral=0.5)
        self.assertFalse(mascara[1])  # col 1 es constante

    def test_conserva_varianza(self):
        X = np.array([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        mascara, var = main.varianza_threshold(X, umbral=0.1)
        self.assertTrue(mascara[0])
        self.assertTrue(mascara[1])


class TestCorrelacion(unittest.TestCase):
    def test_detecta_duplicado(self):
        X = np.random.default_rng(0).normal(size=(50, 3))
        X2 = np.hstack([X, X[:, [0]]])  # col 3 = col 0
        mascara, elim = main.seleccionar_por_correlacion(X2, umbral=0.95)
        # La col 3 (o la 0) debe eliminarse
        self.assertEqual(len(elim), 1)
        self.assertFalse(mascara[list(elim)[0]])

    def test_sin_duplicados(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        mascara, elim = main.seleccionar_por_correlacion(X, umbral=0.95)
        self.assertEqual(len(elim), 0)


class TestMI(unittest.TestCase):
    def test_mi_alta(self):
        # x totalmente predictivo de y
        y = np.array([0, 0, 0, 1, 1, 1])
        x = np.array([0, 0, 0, 1, 1, 1])
        mi = main.mutual_info_simple(x, y)
        self.assertGreater(mi, 0.5)

    def test_mi_baja(self):
        # x e y independientes
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, size=300)
        x = rng.integers(0, 3, size=300)
        mi = main.mutual_info_simple(x, y)
        # Independiente -> MI cercano a 0
        self.assertLess(mi, 0.1)


class TestForward(unittest.TestCase):
    def test_agrega_features(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        # Score = correlacion absoluta promedio con y
        def score(idx, X, y):
            return float(np.mean([abs(np.corrcoef(X[:, i], y)[0, 1]) for i in idx]))
        sel, _ = main.forward_selection(X, y, score, max_features=3)
        self.assertLessEqual(len(sel), 3)
        # Las 2 primeras features deberian ser las mas utiles
        self.assertIn(0, sel)
        self.assertIn(1, sel)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Varianza", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()