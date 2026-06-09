"""Pruebas para 12-ajuste-de-hiperparametros."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestKFolds(unittest.TestCase):
    def test_tamano(self):
        folds = main.k_folds(50, k=5)
        self.assertEqual(len(folds), 5)

    def test_sin_solapamiento(self):
        folds = main.k_folds(50, k=5)
        union = np.concatenate(folds)
        self.assertEqual(len(union), 50)
        self.assertEqual(len(np.unique(union)), 50)


class TestCrossVal(unittest.TestCase):
    def test_regresion(self):
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 2 * X.squeeze() + 1
        def m(Xt, yt, Xe):
            return 2 * Xe.squeeze() + 1
        scores = main.cross_val_score(m, X, y, k=5)
        self.assertEqual(len(scores), 5)
        # Modelo perfecto -> scores muy altos
        self.assertGreater(scores.mean(), 0.99)

    def test_clasificacion(self):
        X = np.random.default_rng(0).normal(size=(50, 2))
        y = (X[:, 0] > 0).astype(int)
        def m(Xt, yt, Xe):
            return (Xe[:, 0] > 0).astype(int)
        scores = main.cross_val_score(m, X, y, k=5)
        self.assertEqual(len(scores), 5)


class TestGridSearch(unittest.TestCase):
    def test_n_resultados(self):
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 2 * X.squeeze() + 1
        def factory(grado=1):
            def m(Xt, yt, Xe):
                coef = np.polyfit(Xt.squeeze(), yt, grado)
                return np.polyval(coef, Xe.squeeze())
            return m
        res = main.grid_search(factory, X, y, {"grado": [1, 2, 3]}, k=5)
        self.assertEqual(len(res), 3)


class TestRandomSearch(unittest.TestCase):
    def test_n_resultados(self):
        X = np.linspace(0, 10, 50).reshape(-1, 1)
        y = 2 * X.squeeze() + 1
        def factory(grado=1):
            def m(Xt, yt, Xe):
                coef = np.polyfit(Xt.squeeze(), yt, grado)
                return np.polyval(coef, Xe.squeeze())
            return m
        dist = {"grado": lambda rng: int(rng.integers(1, 5))}
        res = main.random_search(factory, X, y, dist, n_iter=10, k=5)
        self.assertEqual(len(res), 10)


class TestMejor(unittest.TestCase):
    def test_mejor_mayor_score(self):
        res = [
            {"params": {"a": 1}, "media": 0.5},
            {"params": {"a": 2}, "media": 0.9},
            {"params": {"a": 3}, "media": 0.7},
        ]
        self.assertEqual(main.mejor(res)["params"]["a"], 2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Grid search", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()