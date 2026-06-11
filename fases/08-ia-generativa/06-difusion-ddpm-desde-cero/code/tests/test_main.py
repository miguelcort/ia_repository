"""Pruebas para 06-difusion-ddpm-desde-cero."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSchedules(unittest.TestCase):
    def test_linear(self):
        betas = main.linear_beta_schedule(100)
        self.assertEqual(betas.shape, (100,))
        self.assertAlmostEqual(betas[0], 1e-4)
        self.assertAlmostEqual(betas[-1], 2e-2)

    def test_cosine(self):
        betas = main.cosine_beta_schedule(100)
        self.assertEqual(betas.shape, (100,))
        self.assertTrue((betas > 0).all())
        self.assertTrue((betas < 1).all())


class TestAlphaBar(unittest.TestCase):
    def test_decreciente(self):
        betas = main.linear_beta_schedule(100)
        _, alpha_bar = main.compute_alpha_bar(betas)
        # alpha_bar deberia ser decreciente
        self.assertLess(alpha_bar[-1], alpha_bar[0])
        self.assertAlmostEqual(alpha_bar[0], 1.0 - betas[0], places=6)

    def test_alpha_bar_T_chico(self):
        betas = main.linear_beta_schedule(1000)
        _, alpha_bar = main.compute_alpha_bar(betas)
        # alpha_bar[T] deberia ser muy pequeno
        self.assertLess(alpha_bar[-1], 0.01)


class TestQSample(unittest.TestCase):
    def test_t_cero_es_x0(self):
        betas = main.linear_beta_schedule(100)
        _, alpha_bar = main.compute_alpha_bar(betas)
        x0 = np.random.default_rng(0).standard_normal((2, 4))
        x_t, _ = main.q_sample(x0, 0, alpha_bar, seed=0)
        # t=0: alpha_bar[0] = 1 - beta[0] ~ 1, x_t ~ x0
        np.testing.assert_allclose(x_t, x0, atol=0.02)

    def test_t_T_es_puro_ruido(self):
        betas = main.linear_beta_schedule(1000)
        _, alpha_bar = main.compute_alpha_bar(betas)
        x0 = np.random.default_rng(0).standard_normal((2, 4))
        T = 999
        x_t, _ = main.q_sample(x0, T, alpha_bar, seed=0)
        # alpha_bar[T] deberia ser muy pequeno
        self.assertLess(np.sqrt(alpha_bar[T]), 0.1)

    def test_shape(self):
        betas = main.linear_beta_schedule(100)
        _, alpha_bar = main.compute_alpha_bar(betas)
        x0 = np.random.default_rng(0).standard_normal((2, 4))
        x_t, noise = main.q_sample(x0, 50, alpha_bar, seed=0)
        self.assertEqual(x_t.shape, (2, 4))
        self.assertEqual(noise.shape, (2, 4))


class TestDDPM(unittest.TestCase):
    def test_loss_zero(self):
        noise = np.random.default_rng(0).standard_normal((4,))
        self.assertEqual(main.ddpm_loss(noise, noise), 0.0)

    def test_loss_positiva(self):
        rng = np.random.default_rng(0)
        loss = main.ddpm_loss(rng.standard_normal((4,)),
                               rng.standard_normal((4,)))
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