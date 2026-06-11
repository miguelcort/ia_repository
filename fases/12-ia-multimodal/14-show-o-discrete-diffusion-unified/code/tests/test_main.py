"""Pruebas para 14-show-o-discrete-diffusion-unified."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVocab(unittest.TestCase):
    def test_size(self):
        # 8192 + 50000 + 128 = 58320
        v = main.show_o_vocab()
        self.assertEqual(v, 58320)


class TestAddMask(unittest.TestCase):
    def test_basic(self):
        tokens = np.array([1, 2, 3, 4, 5])
        masked, mask = main.add_mask(tokens, mask_id=999, p_mask=0.5, seed=0)
        # length preserved
        self.assertEqual(masked.shape, tokens.shape)
        self.assertEqual(mask.shape, tokens.shape)

    def test_no_mask(self):
        tokens = np.array([1, 2, 3])
        masked, mask = main.add_mask(tokens, mask_id=999, p_mask=0.0, seed=0)
        np.testing.assert_array_equal(masked, tokens)
        self.assertFalse(mask.any())

    def test_all_mask(self):
        tokens = np.array([1, 2, 3])
        masked, mask = main.add_mask(tokens, mask_id=999, p_mask=1.0, seed=0)
        np.testing.assert_array_equal(masked, [999, 999, 999])
        self.assertTrue(mask.all())


class TestDenoiseStep(unittest.TestCase):
    def test_unmask(self):
        # tokens [mask, 2, mask] -> replace masked
        masked = np.array([999, 2, 999])
        logits = np.zeros((3, 10))
        # predecir id=5 en todas las posiciones
        logits[:, 5] = 10
        out = main.show_o_denoise_step(masked, logits, mask_id=999, t=0)
        # masked positions (0, 2) -> 5; position 1 sigue 2
        self.assertEqual(out[0], 5)
        self.assertEqual(out[1], 2)
        self.assertEqual(out[2], 5)


class TestLoss(unittest.TestCase):
    def test_basic(self):
        logits = np.random.default_rng(0).standard_normal((5, 10))
        targets = np.array([0, 1, 2, 3, 4])
        mask = np.array([True, True, False, True, False])
        loss = main.show_o_loss(logits, targets, mask)
        self.assertGreater(loss, 0)

    def test_no_mask_returns_zero(self):
        logits = np.random.default_rng(0).standard_normal((5, 10))
        targets = np.array([0, 1, 2, 3, 4])
        mask = np.array([False, False, False, False, False])
        loss = main.show_o_loss(logits, targets, mask)
        self.assertEqual(loss, 0.0)


class TestSample(unittest.TestCase):
    def test_image_tokens(self):
        tokens = main.show_o_sample_image(np.array([1, 2]), n_image_tokens=8, n_steps=3)
        self.assertEqual(tokens.shape, (8,))
        for t in tokens:
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, main.show_o_vocab())


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