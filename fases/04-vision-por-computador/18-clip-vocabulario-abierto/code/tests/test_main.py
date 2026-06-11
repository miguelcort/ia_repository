"""Pruebas para 18-clip-vocabulario-abierto."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEmbeddings(unittest.TestCase):
    def test_text_shape(self):
        emb = main.text_encoder_mock("hello")
        self.assertEqual(emb.shape, (512,))

    def test_text_determinista(self):
        e1 = main.text_encoder_mock("cat")
        e2 = main.text_encoder_mock("cat")
        np.testing.assert_array_equal(e1, e2)


class TestNormalize(unittest.TestCase):
    def test_l2(self):
        x = np.array([3.0, 4.0])
        out = main.l2_normalize(x)
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)


class TestSimilarity(unittest.TestCase):
    def test_self_similarity(self):
        x = np.random.default_rng(0).normal(size=(512,))
        # Self-similarity deberia ser 1
        self.assertAlmostEqual(main.clip_similarity(x, x), 1.0, places=4)

    def test_different_similarity(self):
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        # Ortogonales -> similitud 0
        self.assertAlmostEqual(main.clip_similarity(x, y), 0.0)


class TestZeroShot(unittest.TestCase):
    def test_zero_shot(self):
        rng = np.random.default_rng(0)
        img_emb = rng.normal(size=(512,))
        clases = ["a", "b", "c"]
        pred, sims = main.zero_shot_classify(img_emb, clases)
        self.assertIn(pred, [0, 1, 2])
        self.assertEqual(len(sims), 3)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Prediccion", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()