"""Pruebas para 07-aprendizaje-no-supervisado."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestKMeans(unittest.TestCase):
    def test_dos_clusters_separados(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (50, 2)), rng.normal(5, 0.3, (50, 2))])
        centroides, asig = main.kmeans(X, k=2, semilla=0)
        # Cada cluster debe tener ~50 puntos
        sizes = [np.sum(asig == i) for i in range(2)]
        for s in sizes:
            self.assertGreater(s, 30)

    def test_tres_clusters(self):
        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.normal(0, 0.3, (30, 2)),
            rng.normal(5, 0.3, (30, 2)),
            rng.normal(-5, 0.3, (30, 2)),
        ])
        centroides, asig = main.kmeans(X, k=3, semilla=0)
        sizes = [np.sum(asig == i) for i in range(3)]
        for s in sizes:
            self.assertGreater(s, 20)

    def test_reproducible(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 2))
        c1, a1 = main.kmeans(X, k=2, semilla=42)
        c2, a2 = main.kmeans(X, k=2, semilla=42)
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(a1, a2)


class TestInercia(unittest.TestCase):
    def test_inercia_disminuye(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(5, 0.3, (30, 2))])
        _, asig1 = main.kmeans(X, k=1, semilla=0)
        _, asig2 = main.kmeans(X, k=2, semilla=0)
        # Mas clusters = menos inercia
        in1 = main.inercia(X, asig1, [X.mean(axis=0)])
        # Para k=2, recalcular centroides
        from main import kmeans
        _, asig2 = main.kmeans(X, k=2, semilla=0)
        c2, _ = main.kmeans(X, k=2, semilla=0)
        in2 = main.inercia(X, asig2, c2)
        self.assertLess(in2, in1)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Centroides", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()