"""Pruebas para 22-modelos-de-embedding-a-profundidad."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestContrastive(unittest.TestCase):
    def test_loss(self):
        rng = np.random.default_rng(0)
        z1 = main.l2_normalize(rng.normal(size=(4, 64)))
        z2 = main.l2_normalize(rng.normal(size=(4, 64)))
        loss = main.contrastive_loss(z1, z2)
        self.assertGreater(loss, 0.0)


class TestL2Norm(unittest.TestCase):
    def test_norma_uno(self):
        x = np.array([3.0, 4.0])
        out = main.l2_normalize(x)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)


class TestPooling(unittest.TestCase):
    def test_mean_pool(self):
        embeddings = np.array([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
        mask = np.array([[1, 1, 0]])
        pooled = main.mean_pool(embeddings, mask)
        # Solo [1,2] y [3,4]: promedio = [2, 3]
        np.testing.assert_array_almost_equal(pooled[0], [2.0, 3.0])

    def test_cls_pool(self):
        embeddings = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        pooled = main.cls_pool(embeddings)
        np.testing.assert_array_equal(pooled, [[1.0, 2.0]])


class TestHardNegative(unittest.TestCase):
    def test_mining(self):
        rng = np.random.default_rng(0)
        anchor = main.l2_normalize(rng.normal(size=(2, 8)))
        negs = main.l2_normalize(rng.normal(size=(2, 5, 8)))
        hard_idx = main.hard_negative_mining(anchor, anchor, negs, n_hard=2)
        self.assertEqual(hard_idx.shape, (2, 2))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("InfoNCE", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()