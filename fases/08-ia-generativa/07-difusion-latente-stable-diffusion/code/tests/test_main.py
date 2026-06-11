"""Pruebas para 07-difusion-latente-stable-diffusion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVAEEncodeDecode(unittest.TestCase):
    def test_encode_downsample(self):
        x = np.random.default_rng(0).standard_normal((64, 64, 3))
        W_enc = np.random.default_rng(1).standard_normal((3, 4)) * 0.1
        z = main.vae_encode(x, W_enc)
        self.assertEqual(z.shape, (16, 16, 4))

    def test_decode_upsample(self):
        z = np.random.default_rng(0).standard_normal((16, 16, 4))
        W_dec = np.random.default_rng(1).standard_normal((4, 3)) * 0.1
        x = main.vae_decode(z, W_dec)
        self.assertEqual(x.shape, (64, 64, 3))


class TestCFG(unittest.TestCase):
    def test_w_uno_es_conditional(self):
        eps_u = np.array([1.0, 2.0])
        eps_c = np.array([3.0, 4.0])
        out = main.classifier_free_guidance(eps_u, eps_c, w=1.0)
        np.testing.assert_array_equal(out, eps_c)

    def test_w_cero_es_unconditional(self):
        eps_u = np.array([1.0, 2.0])
        eps_c = np.array([3.0, 4.0])
        out = main.classifier_free_guidance(eps_u, eps_c, w=0.0)
        np.testing.assert_array_equal(out, eps_u)

    def test_w_dos_amplifica(self):
        eps_u = np.array([1.0, 2.0])
        eps_c = np.array([3.0, 4.0])
        out = main.classifier_free_guidance(eps_u, eps_c, w=2.0)
        # eps_u + 2*(eps_c - eps_u) = 2*eps_c - eps_u
        expected = 2 * eps_c - eps_u
        np.testing.assert_array_equal(out, expected)


class TestTextConditioning(unittest.TestCase):
    def test_shape(self):
        text_emb = np.random.default_rng(0).standard_normal((1, 77, 768))
        W_proj = np.random.default_rng(1).standard_normal((768, 1024)) * 0.01
        out = main.text_conditioning(text_emb, W_proj)
        self.assertEqual(out.shape, (1, 77, 1024))


class TestSDComponents(unittest.TestCase):
    def test_seis_componentes(self):
        comps = main.sd_components()
        self.assertEqual(len(comps), 6)


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