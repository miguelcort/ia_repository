"""Pruebas para 02-clip-contrastive-pretraining."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestL2Normalize(unittest.TestCase):
    def test_unit_norm(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 8))
        y = main.l2_normalize(x)
        norms = np.linalg.norm(y, axis=-1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-10)

    def test_zero_vector(self):
        x = np.zeros((1, 4))
        y = main.l2_normalize(x)
        # eps prevents div by zero
        self.assertTrue(np.isfinite(y).all())


class TestCosineSim(unittest.TestCase):
    def test_self_similarity(self):
        # vectores i != j, en cosine sim deberia ser < 1
        rng = np.random.default_rng(42)
        x = rng.standard_normal((5, 8))
        sim = main.cosine_sim_matrix(x, x)
        # diagonal = 1 (mismo vector)
        np.testing.assert_allclose(np.diag(sim), np.ones(5), atol=1e-10)
        # off-diagonal < 1 (vectores diferentes)
        for i in range(5):
            for j in range(5):
                if i != j:
                    self.assertLess(sim[i, j], 1.0)

    def test_orthogonal(self):
        a = np.array([[1, 0], [0, 1]], dtype=float)
        b = np.array([[0, 1], [1, 0]], dtype=float)
        sim = main.cosine_sim_matrix(a, b)
        expected = np.array([[0, 1], [1, 0]], dtype=float)
        np.testing.assert_allclose(sim, expected, atol=1e-10)


class TestClipLoss(unittest.TestCase):
    def test_loss_finite(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((8, 16))
        txt = rng.standard_normal((8, 16))
        loss, logits = main.clip_contrastive_loss(img, txt, temperature=0.07)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(logits.shape, (8, 8))

    def test_perfect_alignment_low_loss(self):
        # if img == txt, diagonal >> off-diagonal
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 8))
        loss, _ = main.clip_contrastive_loss(x, x)
        # also: noise
        rng2 = np.random.default_rng(1)
        noise = rng2.standard_normal((4, 8))
        loss2, _ = main.clip_contrastive_loss(x, noise)
        self.assertLess(loss, loss2)

    def test_temperature_scale(self):
        # with identical vectors, lower T -> sharper -> lower loss
        # (positive pair dominates)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 8))
        loss_low, _ = main.clip_contrastive_loss(x, x, temperature=0.01)
        loss_high, _ = main.clip_contrastive_loss(x, x, temperature=1.0)
        self.assertLess(loss_low, loss_high)


class TestClipAccuracy(unittest.TestCase):
    def test_perfect_diag(self):
        # logits: diag = 10, off-diag = 0 -> top-1 perfect
        n = 5
        logits = np.zeros((n, n))
        np.fill_diagonal(logits, 10.0)
        img2t, t2i = main.clip_accuracy(logits, k=1)
        self.assertEqual(img2t, 1.0)
        self.assertEqual(t2i, 1.0)

    def test_zero_off_diag(self):
        # all zeros -> all tied -> top-1 = 1 (tied wins for rank)
        n = 5
        logits = np.zeros((n, n))
        img2t, t2i = main.clip_accuracy(logits, k=1)
        # function uses strict >, so rank = 0 (no value strictly > positive)
        # rank < k=1 -> count = 1 per row
        self.assertEqual(img2t, 1.0)
        self.assertEqual(t2i, 1.0)


class TestZeroShot(unittest.TestCase):
    def test_classifies_best_class(self):
        emb = np.array([1, 0, 0], dtype=float)
        classes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        names = ["A", "B", "C"]
        self.assertEqual(main.zero_shot_classify(emb, classes, names), "A")


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