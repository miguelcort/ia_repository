"""Pruebas para 10-reduccion-de-dimensionalidad."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ

import numpy as np


class TestCentrar(unittest.TestCase):
    def test_centrar_datos_2d(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Xc = main.centrar(X)
        self.assertAlmostEqual(Xc.mean(axis=0)[0], 0.0)
        self.assertAlmostEqual(Xc.mean(axis=0)[1], 0.0)

    def test_centrar_preserva_forma(self):
        X = np.random.default_rng(0).random((10, 3))
        Xc = main.centrar(X)
        self.assertEqual(Xc.shape, X.shape)


class TestCovarianza(unittest.TestCase):
    def test_covarianza_matriz_simetrica(self):
        rng = np.random.default_rng(0)
        X = rng.random((50, 3))
        cov = main.covarianza(X)
        np.testing.assert_allclose(cov, cov.T)

    def test_covarianza_datos_centrados(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        cov = main.covarianza(X)
        # Diagonal debe tener varianzas (>= 0)
        self.assertGreaterEqual(cov[0, 0], 0)
        self.assertGreaterEqual(cov[1, 1], 0)


class TestPCA(unittest.TestCase):
    def test_forma_componentes(self):
        rng = np.random.default_rng(0)
        X = rng.random((50, 5))
        comp, _ = main.pca(X, k=2)
        self.assertEqual(comp.shape, (5, 2))

    def test_forma_proyectados(self):
        rng = np.random.default_rng(0)
        X = rng.random((50, 5))
        _, proy = main.pca(X, k=2)
        self.assertEqual(proy.shape, (50, 2))

    def test_reduce_dimensionalidad(self):
        rng = np.random.default_rng(0)
        X = rng.random((100, 4))
        _, proy = main.pca(X, k=2)
        self.assertEqual(proy.shape[1], 2)

    def test_primer_componente_explica_mas_varianza(self):
        rng = np.random.default_rng(0)
        X = rng.random((100, 3))
        comp, _ = main.pca(X, k=3)
        Xc = main.centrar(X)
        cov = main.covarianza(X)
        # Varianza en cada componente = autovalor
        var_1 = np.var(Xc @ comp[:, 0])
        var_2 = np.var(Xc @ comp[:, 1])
        # PCA ordena por varianza decreciente
        self.assertGreaterEqual(var_1, var_2)


class TestVarianzaExplicada(unittest.TestCase):
    def test_suma_es_uno(self):
        autovalores = np.array([3.0, 2.0, 1.0])
        ve = main.varianza_explicada(autovalores)
        # varianza_explicada ordena de mayor a menor, asi que la suma es 1
        self.assertAlmostEqual(ve.sum(), 1.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Covarianza", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
