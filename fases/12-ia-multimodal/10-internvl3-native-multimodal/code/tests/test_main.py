"""Pruebas para 10-internvl3-native-multimodal."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestInternViT(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((448, 448, 3))
        out = main.intern_vit_forward(img, patch_size=14, embed_dim=1024)
        # 32*32=1024 patches + 1 = 1025
        self.assertEqual(out.shape, (1025, 1024))


class TestDynamicTile(unittest.TestCase):
    def test_max_12_tiles(self):
        img = np.random.default_rng(0).standard_normal((1000, 1500, 3))
        tiles = main.intern_dynamic_tile(img, max_tiles=12)
        nh, nw = None, None
        # find grid
        n = tiles.shape[0]
        # could be 12
        self.assertLessEqual(n, 12)


class TestPixelShuffle(unittest.TestCase):
    def test_reduce_tokens(self):
        feats = np.random.default_rng(0).standard_normal((100, 16))
        out = main.pixel_shuffle_reshape(feats, scale=2)
        # 100/4 = 25 tokens, dim 16*4=64 -> 64
        self.assertEqual(out.shape, (25, 64))


class TestMLPProjector(unittest.TestCase):
    def test_basic(self):
        rng = np.random.default_rng(0)
        feats = rng.standard_normal((10, 16))
        mlp = [(rng.standard_normal((16, 16)) * 0.1, np.zeros(16))]
        out = main.intern_mlp_projector(feats, mlp, target_dim=16)
        self.assertEqual(out.shape, (10, 16))

    def test_dim_change(self):
        rng = np.random.default_rng(0)
        feats = rng.standard_normal((10, 16))
        mlp = [(rng.standard_normal((16, 32)) * 0.1, np.zeros(32))]
        out = main.intern_mlp_projector(feats, mlp, target_dim=32)
        self.assertEqual(out.shape, (10, 32))


class TestInternVL3Forward(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        img = rng.standard_normal((448, 448, 3))
        mlp = [(rng.standard_normal((1024 * 4, 1024)) * 0.02, np.zeros(1024)),
               (rng.standard_normal((1024, 4096)) * 0.02, np.zeros(4096))]
        out = main.internvl3_forward(img, mlp, target_dim=4096, max_tiles=4)
        self.assertEqual(out.shape[1], 4096)


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