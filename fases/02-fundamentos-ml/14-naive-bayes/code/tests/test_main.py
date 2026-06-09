"""Pruebas para 14-naive-bayes."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestGaussianNB(unittest.TestCase):
    def test_fit_y_predict(self):
        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.normal(loc=[-1], scale=0.5, size=(30, 1)),
            rng.normal(loc=[1], scale=0.5, size=(30, 1)),
        ])
        y = np.array([0] * 30 + [1] * 30)
        m = main.GaussianNB()
        m.fit(X, y)
        y_pred = m.predict(X)
        # Accuracy deberia ser alta en datos bien separados
        self.assertGreater(main.accuracy(y, y_pred), 0.85)

    def test_mu_y_sigma(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([0, 0, 1, 1])
        m = main.GaussianNB()
        m.fit(X, y)
        # 2 clases
        self.assertEqual(len(m.clases), 2)
        # mu clase 0: 1.5
        self.assertAlmostEqual(m.mu[0, 0], 1.5)
        # mu clase 1: 3.5
        self.assertAlmostEqual(m.mu[1, 0], 3.5)


class TestMultinomialNB(unittest.TestCase):
    def test_fit_y_predict(self):
        rng = np.random.default_rng(0)
        X = rng.integers(0, 5, size=(50, 3))
        y = (X[:, 0] + X[:, 1] > 4).astype(int)
        m = main.MultinomialNB()
        m.fit(X, y)
        y_pred = m.predict(X)
        # Mayor que azar
        self.assertGreater(main.accuracy(y, y_pred), 0.5)

    def test_smoothing(self):
        # Una clase con cero observaciones de una feature
        X = np.array([[1, 0], [0, 1], [2, 0], [0, 2]])
        y = np.array([0, 0, 1, 1])
        m = main.MultinomialNB(alpha=1.0)
        m.fit(X, y)
        # Sin division por cero
        self.assertFalse(np.isnan(m.log_p_feature).any())


class TestAccuracy(unittest.TestCase):
    def test_perfecto(self):
        y = np.array([0, 1, 0, 1])
        self.assertEqual(main.accuracy(y, y), 1.0)

    def test_cero(self):
        y = np.array([0, 0, 0])
        y_p = np.array([1, 1, 1])
        self.assertEqual(main.accuracy(y, y_p), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("GaussianNB", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()