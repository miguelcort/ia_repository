"""Pruebas para 02-autoencoders-y-vae."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestReparameterize(unittest.TestCase):
    def test_shape(self):
        mu = np.zeros((3, 4))
        logvar = np.zeros((3, 4))
        z = main.reparameterize(mu, logvar, seed=0)
        self.assertEqual(z.shape, (3, 4))

    def test_mu_cero_y_logvar_cero_es_identidad(self):
        # Si mu=0, logvar=0, z = 0 + 1*eps = eps ~ N(0,1)
        mu = np.zeros((1000, 1))
        logvar = np.zeros((1000, 1))
        z = main.reparameterize(mu, logvar, seed=42)
        np.testing.assert_allclose(z.mean(), 0.0, atol=0.1)
        np.testing.assert_allclose(z.std(), 1.0, atol=0.1)

    def test_diferente_seed_diferente_sample(self):
        mu = np.zeros((5, 4))
        logvar = np.zeros((5, 4))
        z1 = main.reparameterize(mu, logvar, seed=0)
        z2 = main.reparameterize(mu, logvar, seed=1)
        self.assertFalse(np.allclose(z1, z2))


class TestKL(unittest.TestCase):
    def test_kl_cero_si_prior(self):
        # mu=0, logvar=0 -> KL=0
        kl = main.kl_divergence(np.zeros((3, 4)), np.zeros((3, 4)))
        self.assertAlmostEqual(kl, 0.0, places=6)

    def test_kl_positivo_si_desviado(self):
        mu = np.array([[1.0, 2.0]])
        kl = main.kl_divergence(mu, np.zeros((1, 2)))
        self.assertGreater(kl, 0)


class TestReconLoss(unittest.TestCase):
    def test_zero_si_identico(self):
        x = np.random.default_rng(0).standard_normal((3, 4))
        loss = main.reconstruction_loss(x, x)
        self.assertEqual(loss, 0.0)

    def test_aumenta_con_error(self):
        x = np.zeros((1, 4))
        l1 = main.reconstruction_loss(x, np.array([[0.1, 0.1, 0.1, 0.1]]))
        l2 = main.reconstruction_loss(x, np.array([[0.5, 0.5, 0.5, 0.5]]))
        self.assertGreater(l2, l1)


class TestVAELoss(unittest.TestCase):
    def test_loss_combinada(self):
        x = np.random.default_rng(0).standard_normal((2, 4))
        x_recon = x + 0.1
        mu = np.zeros((2, 4))
        logvar = np.zeros((2, 4))
        total, recon, kl = main.vae_loss(x, x_recon, mu, logvar)
        self.assertGreater(recon, 0)
        self.assertEqual(kl, 0.0)
        self.assertEqual(total, recon)


class TestForward(unittest.TestCase):
    def test_shapes(self):
        n, d_in, latent = 3, 8, 4
        rng = np.random.default_rng(0)
        x = rng.standard_normal((n, d_in)) * 0.5
        # Init weights
        W_enc = rng.standard_normal((d_in, 16)) * 0.1
        b_enc = np.zeros(16)
        W_mu = rng.standard_normal((16, latent)) * 0.1
        b_mu = np.zeros(latent)
        W_logvar = rng.standard_normal((16, latent)) * 0.1
        b_logvar = np.zeros(latent)
        W_dec = rng.standard_normal((latent, d_in)) * 0.1
        b_dec = np.zeros(d_in)
        x_recon, mu, logvar, z = main.vae_forward(
            x, W_enc, b_enc, W_mu, b_mu, W_logvar, b_logvar, W_dec, b_dec)
        self.assertEqual(x_recon.shape, (n, d_in))
        self.assertEqual(mu.shape, (n, latent))
        self.assertEqual(z.shape, (n, latent))
        # Sigmoid -> range [0, 1]
        self.assertGreaterEqual(x_recon.min(), 0)
        self.assertLessEqual(x_recon.max(), 1.0 + 1e-6)


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