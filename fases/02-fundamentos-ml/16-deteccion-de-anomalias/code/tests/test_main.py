"""Pruebas para 16-deteccion-de-anomalias."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestZScore(unittest.TestCase):
    def test_detecta_outlier(self):
        # Z-score robusto (mediana + MAD)
        x = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1e6])
        anom = main.zscore_robusto(x, umbral=3.5)
        self.assertTrue(anom[6])

    def test_sin_outliers(self):
        x = np.array([1.0, 2.0, 1.5, 2.5, 1.8])
        anom = main.zscore_anomalias(x, umbral=3.0)
        self.assertFalse(anom.any())

    def test_constante(self):
        x = np.ones(10) * 5.0
        anom = main.zscore_anomalias(x)
        self.assertFalse(anom.any())

    def test_robusto_constante(self):
        x = np.ones(10) * 5.0
        anom = main.zscore_robusto(x)
        self.assertFalse(anom.any())


class TestIQR(unittest.TestCase):
    def test_detecta_extremo(self):
        x = np.array([1, 2, 3, 4, 5, 100])
        anom = main.iqr_anomalias(x)
        self.assertTrue(anom[5])

    def test_sin_outliers(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        anom = main.iqr_anomalias(x)
        self.assertFalse(anom.any())


class TestMahalanobis(unittest.TestCase):
    def test_outlier_far(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        X[0] = [10, 10]
        anom, dist = main.mahalanobis_anomalias(X, percentil=95)
        # El outlier debe estar en la cola alta
        self.assertGreater(dist[0], dist[1:].mean())

    def test_2d_shape(self):
        X = np.random.default_rng(0).normal(size=(20, 3))
        anom, dist = main.mahalanobis_anomalias(X)
        self.assertEqual(len(anom), 20)
        self.assertEqual(len(dist), 20)


class TestIsolation(unittest.TestCase):
    def test_devuelve_score(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        scores = main.isolation_score_simple(X, n_arboles=5)
        self.assertEqual(len(scores), 50)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("anomalias", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()