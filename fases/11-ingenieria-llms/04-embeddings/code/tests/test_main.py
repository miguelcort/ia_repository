"""Pruebas para 04-embeddings."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestNormalize(unittest.TestCase):
    def test_norm_uno(self):
        v = np.array([3.0, 4.0])
        n = main.normalize(v)
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-6)


class TestSimilarity(unittest.TestCase):
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(main.cosine_similarity(a, a), 1.0, places=4)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(main.cosine_similarity(a, b), 0.0, places=4)

    def test_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        self.assertAlmostEqual(main.cosine_similarity(a, b), -1.0, places=4)


class TestTopK(unittest.TestCase):
    def test_basic(self):
        # 3 embeddings, query identical a index 0
        q = np.array([1.0, 0.0])
        c = [q, np.array([0.0, 1.0]), np.array([0.0, 0.0])]
        top = main.top_k_similar(q, c, k=2)
        # El primer resultado deberia ser index 0
        self.assertEqual(top[0][0], 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()