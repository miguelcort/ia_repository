"""Pruebas para 13-transfusion-autoregressive-diffusion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestTextTokenize(unittest.TestCase):
    def test_basic(self):
        ids = main.text_token_ids("hello", vocab_size=100)
        self.assertEqual(ids.shape, (5,))
        for i in ids:
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, 100)


class TestDiffusionForward(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((32, 32, 3))
        out = main.diffusion_forward(img, t=10, model=None, sigma_schedule=None)
        self.assertEqual(out.shape, img.shape)


class TestDenoiseStep(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((32, 32, 3))
        noise = np.random.default_rng(1).standard_normal((32, 32, 3))
        out = main.diffusion_denoise_step(img, t=10, predicted_noise=noise,
                                            alpha_t=0.9, sigma_t=0.1)
        self.assertEqual(out.shape, img.shape)


class TestAddNoise(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((32, 32, 3))
        schedule = np.linspace(0.01, 0.99, 100)
        out = main.add_noise(img, t=10, noise_schedule=schedule)
        self.assertEqual(out.shape, img.shape)


class TestLosses(unittest.TestCase):
    def test_text_loss(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((5, 10))
        targets = np.array([0, 1, 2, 3, 4])
        loss = main.transfusion_loss_text(logits, targets)
        self.assertGreater(loss, 0)

    def test_image_loss(self):
        rng = np.random.default_rng(0)
        noisy = rng.standard_normal((4, 4, 3))
        pred = rng.standard_normal((4, 4, 3))
        true = rng.standard_normal((4, 4, 3))
        loss = main.transfusion_loss_image(noisy, pred, true)
        self.assertGreater(loss, 0)
        # perfect prediction -> 0
        loss0 = main.transfusion_loss_image(noisy, true, true)
        self.assertAlmostEqual(loss0, 0.0, places=8)


class TestCombinedLoss(unittest.TestCase):
    def test_step(self):
        rng = np.random.default_rng(0)
        text_logits = rng.standard_normal((5, 10))
        targets = np.array([0, 1, 2, 3, 4])
        noisy = rng.standard_normal((4, 4, 3))
        pred = rng.standard_normal((4, 4, 3))
        true = rng.standard_normal((4, 4, 3))
        loss = main.transfusion_step(text_logits, targets, noisy, pred, true)
        self.assertGreater(loss, 0)


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