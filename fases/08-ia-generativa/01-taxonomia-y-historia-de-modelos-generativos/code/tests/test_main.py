"""Pruebas para 01-taxonomia-y-historia-de-modelos-generativos."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestELBO(unittest.TestCase):
    def test_suma(self):
        self.assertAlmostEqual(main.elbo_loss(0.5, 0.3), 0.8, places=6)

    def test_no_negativo_si_terminos_positivos(self):
        self.assertGreater(main.elbo_loss(0.5, 0.3), 0)


class TestGANLosses(unittest.TestCase):
    def test_d_loss_positivo(self):
        d, g = main.gan_losses(d_real_logits=1.0, d_fake_logits=-1.0)
        self.assertGreater(d, 0)

    def test_g_loss_positivo(self):
        d, g = main.gan_losses(d_real_logits=1.0, d_fake_logits=-1.0)
        self.assertGreater(g, 0)

    def test_perfect_discriminator_d_loss_bajo(self):
        # Si D(x_real) = +inf, D(x_fake) = -inf -> D_loss ~= 0
        d, g = main.gan_losses(d_real_logits=20.0, d_fake_logits=-20.0)
        self.assertLess(d, 0.01)


class TestDiffusionLoss(unittest.TestCase):
    def test_zero_si_perfecto(self):
        noise = np.random.default_rng(0).standard_normal((4, 4))
        loss = main.diffusion_loss(noise, noise)
        self.assertEqual(loss, 0.0)

    def test_positivo(self):
        rng = np.random.default_rng(0)
        loss = main.diffusion_loss(rng.standard_normal((4, 4)),
                                     rng.standard_normal((4, 4)))
        self.assertGreater(loss, 0)


class TestFlowMatching(unittest.TestCase):
    def test_zero_si_perfecto(self):
        v = np.random.default_rng(0).standard_normal((4, 4))
        self.assertEqual(main.flow_matching_loss(v, v), 0.0)


class TestTaxonomy(unittest.TestCase):
    def test_seis_familias(self):
        families = main.taxonomy_summary()
        self.assertEqual(len(families), 6)
        names = {n for n, _ in families}
        for f in ["VAE", "GAN", "Diffusion"]:
            self.assertIn(f, names)


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