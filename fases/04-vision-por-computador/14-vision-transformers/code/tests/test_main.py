"""Pruebas para 14-vision-transformers."""
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
    def test_patches_count(self):
        img = np.random.default_rng(0).normal(size=(32, 32, 3))
        patches = main.dividir_en_patches(img, patch_size=16)
        # 32/16 = 2 por dim, 2*2 = 4 patches
        self.assertEqual(patches.shape, (4, 16 * 16 * 3))

    def test_patches_224(self):
        img = np.random.default_rng(0).normal(size=(224, 224, 3))
        patches = main.dividir_en_patches(img, patch_size=16)
        # 14*14 = 196 patches
        self.assertEqual(patches.shape, (196, 16 * 16 * 3))


class TestEmbedding(unittest.TestCase):
    def test_embedding_shape(self):
        patches = np.random.default_rng(0).normal(size=(4, 768))
        emb = main.patch_embedding(patches, dim_embedding=256)
        self.assertEqual(emb.shape, (4, 256))

    def test_pe_shape(self):
        pe = main.positional_embedding(n_patches=10, dim_embedding=64)
        self.assertEqual(pe.shape, (10, 64))


class TestAttention(unittest.TestCase):
    def test_attention_shape(self):
        x = np.random.default_rng(0).normal(size=(10, 768))
        out = main.self_attention_simple(x, dim_head=64)
        self.assertEqual(out.shape, (10, 64))


class TestClasificacion(unittest.TestCase):
    def test_head(self):
        features = np.random.default_rng(0).normal(size=(10, 768))
        logits = main.classification_head(features, n_clases=100)
        self.assertEqual(logits.shape, (100,))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Logits", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()