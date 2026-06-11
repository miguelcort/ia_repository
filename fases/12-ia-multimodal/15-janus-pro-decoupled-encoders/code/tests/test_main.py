"""Pruebas para 15-janus-pro-decoupled-encoders."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestUnderstandingEncoder(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((224, 224, 3))
        feats = main.janus_understanding_encoder(img, embed_dim=1152)
        # 16*16 + 1 = 257
        self.assertEqual(feats.shape, (257, 1152))


class TestGenerationEncoder(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((256, 256, 3))
        codes = main.janus_generation_encoder(img, vqvae=32768, patch_size=16)
        # 16*16 = 256
        self.assertEqual(codes.shape, (256,))
        self.assertLess(codes.max(), 32768)


class TestRoute(unittest.TestCase):
    def test_understanding(self):
        r = main.janus_route_modality("understanding", "image")
        self.assertEqual(r, "siglip")

    def test_generation(self):
        r = main.janus_route_modality("generation", "image")
        self.assertEqual(r, "vqvae")


class TestJanusForward(unittest.TestCase):
    def test_understanding_mode(self):
        img = np.random.default_rng(0).standard_normal((224, 224, 3))
        text = np.array([1, 2, 3])
        out = main.janus_forward(img, text, mode="understanding", embed_dim=4096)
        # 257 img + 3 text = 260
        self.assertEqual(out.shape, (260, 4096))

    def test_generation_mode(self):
        img = np.random.default_rng(0).standard_normal((256, 256, 3))
        text = np.array([1, 2, 3])
        out = main.janus_forward(img, text, mode="generation", embed_dim=4096)
        # 256 img + 3 text = 259
        self.assertEqual(out.shape, (259, 4096))


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