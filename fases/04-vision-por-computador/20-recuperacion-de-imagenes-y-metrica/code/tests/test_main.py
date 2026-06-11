"""Pruebas para 20-recuperacion-de-imagenes-y-metrica."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEncoder(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).normal(size=(10, 10, 3))
        e = main.encoder_mock(img, dim=128, semilla=0)
        self.assertEqual(e.shape, (128,))


class TestIndice(unittest.TestCase):
    def test_shape(self):
        embeddings = [np.random.default_rng(i).normal(size=(128,)) for i in range(10)]
        idx = main.construir_indice(embeddings)
        self.assertEqual(idx.shape, (10, 128))

    def test_normalizado(self):
        embeddings = [np.random.default_rng(i).normal(size=(128,)) for i in range(5)]
        idx = main.construir_indice(embeddings)
        # Cada fila norma 1
        for i in range(5):
            self.assertAlmostEqual(float(np.linalg.norm(idx[i])), 1.0, places=5)


class TestBusqueda(unittest.TestCase):
    def test_top_k_shape(self):
        embeddings = [np.random.default_rng(i).normal(size=(128,)) for i in range(20)]
        idx = main.construir_indice(embeddings)
        query = np.random.default_rng(99).normal(size=(128,))
        top, sims = main.buscar_por_similitud(query, idx, top_k=5)
        self.assertEqual(len(top), 5)
        self.assertEqual(len(sims), 5)
        # Sims ordenados descendente
        for i in range(len(sims) - 1):
            self.assertGreaterEqual(sims[i], sims[i + 1])


class TestMetrics(unittest.TestCase):
    def test_recall_at_k(self):
        # 3 relevantes, top-2 incluye 1
        relevantes = [1, 2, 3]
        retrieved = [0, 1, 4, 5]
        r = main.recall_at_k(relevantes, retrieved, k=2)
        self.assertAlmostEqual(r, 1 / 3)

    def test_precision_at_k(self):
        relevantes = [1, 2, 3]
        retrieved = [0, 1, 4, 5]
        # top-2: [0, 1], 1 es relevante -> 1/2
        p = main.precision_at_k(relevantes, retrieved, k=2)
        self.assertAlmostEqual(p, 0.5)

    def test_map(self):
        rels = [[0, 2]]
        rankings = [[0, 1, 2, 3]]
        # AP: en pos 0 hit (p=1.0), pos 2 hit (p=2/3). AP = (1 + 0.667)/2 = 0.833
        self.assertAlmostEqual(main.mean_average_precision(rels, rankings), (1 + 2/3) / 2, places=2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Top-5", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()