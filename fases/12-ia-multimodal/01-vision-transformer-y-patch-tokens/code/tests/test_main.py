"""Pruebas para 01-vision-transformer-y-patch-tokens."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPatches(unittest.TestCase):
    def test_basic(self):
        img = np.random.default_rng(0).standard_normal((32, 32, 3))
        patches = main.image_to_patches(img, patch_size=8)
        # 32/8 = 4, 4*4 = 16 patches
        self.assertEqual(patches.shape[0], 16)
        # 8*8*3 = 192
        self.assertEqual(patches.shape[1], 192)

    def test_vit_standard(self):
        img = np.random.default_rng(0).standard_normal((224, 224, 3))
        patches = main.image_to_patches(img, patch_size=16)
        # 14*14 = 196
        self.assertEqual(patches.shape[0], 196)


class TestPatchEmbed(unittest.TestCase):
    def test_basic(self):
        patches = np.random.default_rng(0).standard_normal((16, 192))
        W = np.random.default_rng(1).standard_normal((192, 64)) * 0.1
        b = np.zeros(64)
        emb = main.patch_embedding(patches, W, b)
        self.assertEqual(emb.shape, (16, 64))


class TestClassToken(unittest.TestCase):
    def test_prepend(self):
        emb = np.random.default_rng(0).standard_normal((4, 8))
        cls = np.ones(8) * 99
        out = main.add_class_token(emb, cls)
        self.assertEqual(out.shape, (5, 8))
        np.testing.assert_array_equal(out[0], cls)


class TestNPatches(unittest.TestCase):
    def test_vit_b_16(self):
        # 224/16 = 14, 14*14 = 196
        self.assertEqual(main.n_patches(224, 16), 196)

    def test_vit_l_14(self):
        # 224/14 = 16, 16*16 = 256
        self.assertEqual(main.n_patches(224, 14), 256)


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