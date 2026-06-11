"""Pruebas para 04-gans-condicionales-pix2pix."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestConcatCondition(unittest.TestCase):
    def test_concat(self):
        z = np.zeros((3, 4))
        c = np.zeros((3, 2))
        out = main.concat_condition(z, c)
        self.assertEqual(out.shape, (3, 6))

    def test_concat_preserva_datos(self):
        z = np.array([[1, 2]])
        c = np.array([[3, 4, 5]])
        out = main.concat_condition(z, c)
        np.testing.assert_array_equal(out, [[1, 2, 3, 4, 5]])


class TestCGANGenerator(unittest.TestCase):
    def test_shape(self):
        z = np.random.default_rng(0).standard_normal((3, 4))
        c = np.random.default_rng(1).standard_normal((3, 2))
        W1 = np.random.default_rng(2).standard_normal((6, 8)) * 0.1
        b1 = np.zeros(8)
        W2 = np.random.default_rng(3).standard_normal((8, 5)) * 0.1
        b2 = np.zeros(5)
        out = main.conditional_generator(z, c, W1, b1, W2, b2)
        self.assertEqual(out.shape, (3, 5))


class TestCGANDiscriminator(unittest.TestCase):
    def test_shape(self):
        x = np.random.default_rng(0).standard_normal((3, 5))
        c = np.random.default_rng(1).standard_normal((3, 2))
        W1 = np.random.default_rng(2).standard_normal((7, 8)) * 0.1
        b1 = np.zeros(8)
        W2 = np.random.default_rng(3).standard_normal((8, 1)) * 0.1
        b2 = np.zeros(1)
        out = main.conditional_discriminator(x, c, W1, b1, W2, b2)
        self.assertEqual(out.shape, (3, 1))


class TestProjectionDiscriminator(unittest.TestCase):
    def test_shape(self):
        x = np.random.default_rng(0).standard_normal((3, 5))
        c = np.random.default_rng(1).standard_normal((3, 2))
        W_proj = np.random.default_rng(2).standard_normal((2, 1)) * 0.1
        W1 = np.random.default_rng(3).standard_normal((5, 8)) * 0.1
        b1 = np.zeros(8)
        W2 = np.random.default_rng(4).standard_normal((8, 1)) * 0.1
        b2 = np.zeros(1)
        out = main.projection_discriminator(x, c, W_proj, W1, b1, W2, b2)
        self.assertEqual(out.shape, (3, 1))


class TestPix2PixL1(unittest.TestCase):
    def test_zero_si_identico(self):
        x = np.random.default_rng(0).standard_normal((3, 4))
        self.assertEqual(main.pix2pix_l1_loss(x, x), 0.0)

    def test_aumenta_con_diferencia(self):
        x = np.zeros((1, 4))
        l1 = main.pix2pix_l1_loss(x, np.array([[0.1, 0.1, 0.1, 0.1]]))
        l2 = main.pix2pix_l1_loss(x, np.array([[0.5, 0.5, 0.5, 0.5]]))
        self.assertGreater(l2, l1)


class TestCGANLosses(unittest.TestCase):
    def test_d_loss_positivo(self):
        d_real = np.array([1.0, 1.0])
        d_fake = np.array([-1.0, -1.0])
        self.assertGreater(main.cgan_d_loss(d_real, d_fake), 0)

    def test_g_loss_no_saturating(self):
        d_fake = np.array([1.0, 1.0])
        g_loss = main.cgan_g_loss(d_fake)
        self.assertAlmostEqual(g_loss, -1.0, places=5)


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