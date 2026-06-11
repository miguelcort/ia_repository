"""Pruebas para 10-generacion-de-imagenes-con-difusion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSchedule(unittest.TestCase):
    def test_beta_shape(self):
        b = main.beta_schedule(1000)
        self.assertEqual(b.shape, (1000,))

    def test_beta_rango(self):
        b = main.beta_schedule(100)
        self.assertEqual(b[0], 1e-4)
        self.assertAlmostEqual(b[-1], 0.02, places=5)


class TestPrecompute(unittest.TestCase):
    def test_alpha_bars_decreciente(self):
        b = main.beta_schedule(1000)
        _, ab = main.precompute_diffusion(b)
        # alpha_bar = product(1 - beta) decrece monotono
        for i in range(len(ab) - 1):
            self.assertLessEqual(ab[i + 1], ab[i])
        # Termina cerca de 0
        self.assertLess(ab[-1], 0.1)


class TestForward(unittest.TestCase):
    def test_t0_sin_ruido(self):
        b = main.beta_schedule(1000)
        _, ab = main.precompute_diffusion(b)
        x0 = np.ones((4, 4))
        x_t, ruido = main.forward_diffusion(x0, 0, ab, semilla=42)
        # En t=0, alpha_bar ~= 1, x_t ~= x0 + ruido pequeno
        # sqrt(ab[0]) * 1 + sqrt(1-ab[0]) * ruido ~= 1*1 + sqrt(1-0.9999)*ruido
        # Aprox x0 + ruido * 0.01
        np.testing.assert_array_almost_equal(x_t, np.sqrt(ab[0]) * x0 + np.sqrt(1 - ab[0]) * ruido, decimal=5)

    def test_t999_mas_ruido(self):
        b = main.beta_schedule(1000)
        _, ab = main.precompute_diffusion(b)
        x0 = np.ones((4, 4))
        x_t, _ = main.forward_diffusion(x0, 999, ab, semilla=42)
        # En t=999, alpha_bar es muy pequeno, x_t ~= ruido puro (std cercano a 1)
        self.assertGreater(x_t.std(), 0.8)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("alpha_bars", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()