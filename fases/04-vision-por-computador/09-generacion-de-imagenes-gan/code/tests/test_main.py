"""Pruebas para 09-generacion-de-imagenes-gan."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestRuido(unittest.TestCase):
    def test_shape(self):
        z = main.ruido(100, 16)
        self.assertEqual(z.shape, (16, 100))

    def test_distribucion(self):
        z = main.ruido(100, 10000)
        # Media cercana a 0
        self.assertAlmostEqual(float(z.mean()), 0.0, places=1)
        # Std cercano a 1
        self.assertAlmostEqual(float(z.std()), 1.0, places=1)


class TestModelo(unittest.TestCase):
    def test_generador(self):
        z = main.ruido(50, 8)
        out = main.generador_minimo(z, salida_dim=784)
        self.assertEqual(out.shape, (8, 784))
        # tanh en [-1, 1]
        self.assertGreaterEqual(out.min(), -1.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_discriminador(self):
        x = np.random.default_rng(0).normal(size=(8, 784))
        out = main.discriminador_minimo(x)
        self.assertEqual(out.shape, (8, 1))


class TestLosses(unittest.TestCase):
    def test_grad_disc(self):
        x_real = np.random.default_rng(0).normal(size=(4, 784))
        x_fake = np.random.default_rng(1).normal(size=(4, 784))
        loss = main.grad_discriminador(x_real, x_fake)
        self.assertGreater(loss, 0.0)

    def test_grad_gen(self):
        z = main.ruido(100, 4)
        loss = main.grad_generador(z)
        self.assertGreater(loss, 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Loss", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()