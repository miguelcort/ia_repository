"""Pruebas para 09-vision-transformers."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestImageToPatches(unittest.TestCase):
    def test_numero_patches(self):
        img = np.random.default_rng(0).standard_normal((32, 32, 3))
        patches = main.image_to_patches(img, patch_size=8)
        # 32/8 = 4, 4x4 = 16 patches
        self.assertEqual(patches.shape[0], 16)

    def test_patch_dim(self):
        img = np.random.default_rng(0).standard_normal((32, 32, 3))
        patches = main.image_to_patches(img, patch_size=8)
        # 8*8*3 = 192
        self.assertEqual(patches.shape[1], 8 * 8 * 3)

    def test_patches_reconstruibles(self):
        # patches son subsets de la imagen
        img = np.arange(16).reshape(4, 4, 1).astype(float)
        patches = main.image_to_patches(img, patch_size=2)
        # 4 patches de 4*1 = 4 dim
        # patch 0: top-left 2x2 = [0,1,4,5]
        np.testing.assert_array_equal(patches[0], [0, 1, 4, 5])


class TestPatchEmbeddings(unittest.TestCase):
    def test_shape(self):
        patches = np.random.default_rng(0).standard_normal((16, 192))
        W = np.random.default_rng(1).standard_normal((64, 192)) * 0.02
        b = np.zeros(64)
        out = main.patch_embeddings(patches, W, b)
        self.assertEqual(out.shape, (16, 64))


class TestClassToken(unittest.TestCase):
    def test_prepend(self):
        emb = np.random.default_rng(0).standard_normal((4, 8))
        cls = np.ones(8) * 99
        out = main.add_class_token(emb, cls)
        self.assertEqual(out.shape, (5, 8))
        np.testing.assert_array_equal(out[0], cls)
        np.testing.assert_array_equal(out[1:], emb)


class TestViT(unittest.TestCase):
    def test_classifier_usa_cls(self):
        emb = np.random.default_rng(0).standard_normal((5, 8))
        W = np.random.default_rng(1).standard_normal((3, 8))
        out = main.vit_classifier(emb, W)
        self.assertEqual(out.shape, (3,))
        expected = emb[0] @ W.T
        np.testing.assert_allclose(out, expected)


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