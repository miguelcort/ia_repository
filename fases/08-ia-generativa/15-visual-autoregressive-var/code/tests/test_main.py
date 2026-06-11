"""Pruebas para 15-visual-autoregressive-var."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPatchify(unittest.TestCase):
    def test_numero_patches(self):
        img = np.random.default_rng(0).standard_normal((8, 8, 3))
        tokens = main.patchify_to_tokens(img, patch_size=2)
        # 4x4 = 16 patches
        self.assertEqual(tokens.shape[0], 16)

    def test_dim(self):
        img = np.random.default_rng(0).standard_normal((8, 8, 3))
        tokens = main.patchify_to_tokens(img, patch_size=2)
        # 2*2*3 = 12
        self.assertEqual(tokens.shape[1], 12)


class TestQuantize(unittest.TestCase):
    def test_cuenta_codigos(self):
        tokens = np.random.default_rng(0).standard_normal((4, 8))
        codebook = np.random.default_rng(1).standard_normal((10, 8))
        idx = main.var_quantize(tokens, codebook)
        self.assertEqual(idx.shape, (4,))
        # Cada idx deberia ser 0..9
        self.assertGreaterEqual(idx.min(), 0)
        self.assertLess(idx.max(), 10)


class TestPredictScale(unittest.TestCase):
    def test_shape(self):
        prev = [np.zeros((1,)), np.zeros((4,))]
        def model_fn(x, scale_idx):
            return np.zeros((4, 16))  # 4 tokens, 16 codebook size
        tokens = main.var_predict_scale(model_fn, prev, scale_idx=2, codebook_size=16)
        self.assertEqual(tokens.shape, (4,))


class TestNextScale(unittest.TestCase):
    def test_primera_es_1x1(self):
        out = main.next_scale_prediction([], n_scales=10, target_res=64)
        self.assertEqual(out.shape, (1, 1))

    def test_crece(self):
        s1 = main.next_scale_prediction([], n_scales=10, target_res=64)
        s2 = main.next_scale_prediction([s1], n_scales=10, target_res=64)
        self.assertGreater(s2.shape[0], s1.shape[0])


class TestVAR(unittest.TestCase):
    def test_seis_keys(self):
        comps = main.var_vs_diffusion()
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