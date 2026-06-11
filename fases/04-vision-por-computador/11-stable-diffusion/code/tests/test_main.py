"""Pruebas para 11-stable-diffusion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestTextEncoder(unittest.TestCase):
    def test_shape(self):
        emb = main.simulador_text_encoder("hello world")
        self.assertEqual(emb.shape, (1, 768))

    def test_determinista(self):
        emb1 = main.simulador_text_encoder("test")
        emb2 = main.simulador_text_encoder("test")
        np.testing.assert_array_equal(emb1, emb2)


class TestCFG(unittest.TestCase):
    def test_w_cero(self):
        # w=0 -> solo uncond
        eps_cond = np.array([1.0, 2.0])
        eps_uncond = np.array([0.5, 1.0])
        out = main.cfg(eps_cond, eps_uncond, w=0)
        np.testing.assert_array_equal(out, eps_uncond)

    def test_w_uno(self):
        # w=1 -> interpolacion simple
        eps_cond = np.array([1.0, 2.0])
        eps_uncond = np.array([0.0, 0.0])
        out = main.cfg(eps_cond, eps_uncond, w=1)
        np.testing.assert_array_equal(out, eps_cond)


class TestVAE(unittest.TestCase):
    def test_encode(self):
        img = np.random.default_rng(0).normal(size=(1, 3, 64, 64))
        lat = main.simulador_vae_encode(img)
        # 64/8 = 8
        self.assertEqual(lat.shape, (1, 4, 8, 8))

    def test_decode(self):
        lat = np.random.default_rng(0).normal(size=(1, 4, 64, 64))
        img = main.simulador_vae_decode(lat)
        # 64*8 = 512
        self.assertEqual(img.shape, (1, 3, 512, 512))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Imagen", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()