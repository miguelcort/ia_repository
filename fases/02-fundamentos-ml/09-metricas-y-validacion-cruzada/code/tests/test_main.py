"""Pruebas para 09-metricas-y-validacion-cruzada."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(main.accuracy(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])), 1.0)
        self.assertEqual(main.accuracy(np.array([0, 1]), np.array([1, 0])), 0.0)

    def test_precision(self):
        # tp=2, fp=0
        self.assertEqual(main.precision(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])), 1.0)
        # tp=1, fp=1 -> 0.5
        self.assertAlmostEqual(main.precision(np.array([1, 0]), np.array([1, 1])), 0.5)

    def test_recall(self):
        # tp=2, fn=0
        self.assertEqual(main.recall(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])), 1.0)
        # tp=1, fn=1 -> 0.5
        self.assertAlmostEqual(main.recall(np.array([1, 1]), np.array([1, 0])), 0.5)

    def test_f1(self):
        # tp=2, fp=0, fn=0 -> f1=1
        self.assertEqual(main.f1(np.array([0, 1, 1, 0]), np.array([0, 1, 1, 0])), 1.0)


class TestKFold(unittest.TestCase):
    def test_k_fold_cubre_todos(self):
        n = 100
        folds = main.k_fold(n, k=5)
        todos = np.concatenate(folds)
        self.assertEqual(len(todos), n)
        self.assertEqual(len(np.unique(todos)), n)

    def test_k_fold_5_tiene_5_folds(self):
        folds = main.k_fold(50, k=5)
        self.assertEqual(len(folds), 5)


class TestCrossVal(unittest.TestCase):
    def test_cv_dummy(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        y = rng.integers(0, 2, 50)
        # Modelo que siempre predice 0
        scores = main.cross_val_score(
            lambda Xt, yt, Xe: np.zeros(len(Xe), dtype=int),
            X, y, k=5,
        )
        # Accuracy ~ 50% (clases balanceadas)
        self.assertAlmostEqual(scores.mean(), 0.5, delta=0.2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Accuracy", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()