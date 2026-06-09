"""Pruebas para 06-knn-y-distancias."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEuclidiana(unittest.TestCase):
    def test_euclidiana_3_4_5(self):
        self.assertAlmostEqual(main.euclidiana(np.array([0, 0]), np.array([3, 4])), 5.0)

    def test_euclidiana_cero(self):
        self.assertEqual(main.euclidiana(np.array([1, 2, 3]), np.array([1, 2, 3])), 0.0)


class TestKNN(unittest.TestCase):
    def test_knn_separables(self):
        X_tr = np.vstack([np.zeros((10, 2)), np.ones((10, 2)) * 10])
        y_tr = np.array([0] * 10 + [1] * 10)
        # Test point cerca de origen
        y_pred = main.knn(X_tr, y_tr, np.array([[0.1, 0.1]]), k=3)
        self.assertEqual(y_pred[0], 0)
        # Test point cerca de (10, 10)
        y_pred = main.knn(X_tr, y_tr, np.array([[9.9, 9.9]]), k=3)
        self.assertEqual(y_pred[0], 1)

    def test_knn_k_1(self):
        # Con k=1, predice el mas cercano
        X_tr = np.array([[0.0, 0.0], [5.0, 5.0]])
        y_tr = np.array([0, 1])
        y_pred = main.knn(X_tr, y_tr, np.array([[0.1, 0.1]]), k=1)
        self.assertEqual(y_pred[0], 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Predicciones", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()