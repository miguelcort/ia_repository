"""Pruebas para 03-gans-generador-y-discriminador."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestGenerator(unittest.TestCase):
    def test_shape(self):
        W1 = np.random.default_rng(0).standard_normal((8, 16)) * 0.1
        b1 = np.zeros(16)
        W2 = np.random.default_rng(1).standard_normal((16, 4)) * 0.1
        b2 = np.zeros(4)
        z = np.random.default_rng(2).standard_normal((3, 8))
        out = main.generator_forward(z, W1, b1, W2, b2)
        self.assertEqual(out.shape, (3, 4))
        # Sigmoid -> [0, 1]
        self.assertGreaterEqual(out.min(), 0)
        self.assertLessEqual(out.max(), 1.0 + 1e-6)


class TestDiscriminator(unittest.TestCase):
    def test_shape(self):
        W1 = np.random.default_rng(0).standard_normal((4, 16)) * 0.1
        b1 = np.zeros(16)
        W2 = np.random.default_rng(1).standard_normal((16, 1)) * 0.1
        b2 = np.zeros(1)
        x = np.random.default_rng(2).standard_normal((3, 4))
        out = main.discriminator_forward(x, W1, b1, W2, b2)
        self.assertEqual(out.shape, (3, 1))


class TestLosses(unittest.TestCase):
    def test_d_loss_no_saturating(self):
        # D_loss = -D(real) + D(fake)
        d_real = np.array([1.0, 1.0])
        d_fake = np.array([-1.0, -1.0])
        d_loss = main.gan_d_loss(d_real, d_fake, "non-saturating")
        # -1 + (-1) = -2
        self.assertAlmostEqual(d_loss, -2.0, places=5)

    def test_g_loss_no_saturating(self):
        d_fake = np.array([1.0, 1.0])
        g_loss = main.gan_g_loss(d_fake, "non-saturating")
        self.assertAlmostEqual(g_loss, -1.0, places=5)

    def test_d_loss_vanilla_positivo(self):
        d_real = np.array([0.5, 0.5])
        d_fake = np.array([-0.5, -0.5])
        d_loss = main.gan_d_loss(d_real, d_fake, "vanilla")
        self.assertGreater(d_loss, 0)


class TestWasserstein(unittest.TestCase):
    def test_wasserstein_loss(self):
        d_real = np.array([1.0, 1.0])
        d_fake = np.array([-1.0, -1.0])
        w = main.wasserstein_loss(d_real, d_fake)
        # D(fake) - D(real) = -1 - 1 = -2
        self.assertAlmostEqual(w, -2.0, places=5)


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