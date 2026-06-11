"""Pruebas para 05-stylegan."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestAffineTransform(unittest.TestCase):
    def test_shape(self):
        z = np.random.default_rng(0).standard_normal((4, 8))
        W = np.random.default_rng(1).standard_normal((8, 16)) * 0.1
        b = np.zeros(16)
        out = main.affine_transform(z, W, b)
        self.assertEqual(out.shape, (4, 16))


class TestAdaIN(unittest.TestCase):
    def test_shape(self):
        x = np.random.default_rng(0).standard_normal((4, 8))
        w_y = np.random.default_rng(1).standard_normal((4, 8))
        w_b = np.random.default_rng(2).standard_normal((4, 8))
        out = main.adain(x, w_y, w_b)
        self.assertEqual(out.shape, (4, 8))

    def test_normalizado(self):
        # Output deberia tener var ~ w_y^2 (sin w)
        x = np.random.default_rng(0).standard_normal((100, 32))
        w_y = np.ones((100, 32))  # No scaling
        w_b = np.zeros((100, 32))
        out = main.adain(x, w_y, w_b)
        # Output deberia tener std ~ 1 (por instancia, pero aqui batch-wise)
        # Verificamos que la normalizacion se aplico
        # Solo verificamos que no es igual a x
        self.assertFalse(np.allclose(out, x))


class TestStyleMixing(unittest.TestCase):
    def test_shape(self):
        w1 = np.random.default_rng(0).standard_normal((5, 8))
        w2 = np.random.default_rng(1).standard_normal((5, 8))
        mixed, layer = main.style_mixing(w1, w2, threshold=3, seed=0)
        self.assertEqual(mixed.shape, (5, 8))


class TestTruncation(unittest.TestCase):
    def test_psi_cero(self):
        w = np.random.default_rng(0).standard_normal((4, 8))
        w_avg = np.zeros(8)
        out = main.truncation_trick(w, psi=0.0, w_avg=w_avg)
        np.testing.assert_allclose(out, np.zeros((4, 8)), atol=1e-9)

    def test_psi_uno_identidad(self):
        w = np.random.default_rng(0).standard_normal((4, 8))
        w_avg = w.mean(axis=0)
        out = main.truncation_trick(w, psi=1.0, w_avg=w_avg)
        np.testing.assert_allclose(out, w, atol=1e-9)

    def test_psi_punto_cinco(self):
        w = np.random.default_rng(0).standard_normal((4, 8))
        w_avg = w.mean(axis=0)
        out = main.truncation_trick(w, psi=0.5, w_avg=w_avg)
        # Promedio entre w y w_avg
        expected = 0.5 * (w + w_avg)
        np.testing.assert_allclose(out, expected, atol=1e-9)


class TestProgressiveGrow(unittest.TestCase):
    def test_layers_crecen(self):
        layers = main.progressive_grow_layers(target_res=16, latent_dim=8)
        self.assertGreater(len(layers), 0)


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