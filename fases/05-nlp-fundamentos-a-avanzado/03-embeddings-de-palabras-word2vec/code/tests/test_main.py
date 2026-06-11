"""Pruebas para 03-embeddings-de-palabras-word2vec."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSkipGram(unittest.TestCase):
    def test_loss_positivo(self):
        V, D = 100, 10
        rng = np.random.default_rng(0)
        target_emb = rng.normal(size=D)
        context_emb = rng.normal(size=D)
        loss = main.skip_gram_step(target_emb, context_emb, V, n_neg=5)
        self.assertGreater(loss, 0.0)


class TestSigmoid(unittest.TestCase):
    def test_rango(self):
        self.assertAlmostEqual(main.sigmoid(0), 0.5)
        self.assertAlmostEqual(main.sigmoid(100), 1.0, places=5)
        self.assertAlmostEqual(main.sigmoid(-100), 0.0, places=5)


class TestSimilarity(unittest.TestCase):
    def test_identidad(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(main.cosine_similarity(v, v), 1.0, places=5)


class TestAnalogia(unittest.TestCase):
    def test_shape(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        c = np.array([0.0, 0.0])
        res = main.analogia(a, b, c)
        self.assertEqual(res.shape, (2,))
        # c + b - a = (-1, 1)
        np.testing.assert_array_almost_equal(res, [-1, 1])


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Loss", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()