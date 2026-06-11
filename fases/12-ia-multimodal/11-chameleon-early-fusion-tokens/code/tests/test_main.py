"""Pruebas para 11-chameleon-early-fusion-tokens."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVQVAE(unittest.TestCase):
    def test_encode(self):
        img = np.random.default_rng(0).standard_normal((256, 256, 3))
        codes = main.vqvae_encode(img, codebook_size=8192, patch_size=16)
        # 16*16 = 256 codes
        self.assertEqual(codes.shape, (256,))
        self.assertGreaterEqual(codes.min(), 0)
        self.assertLess(codes.max(), 8192)

    def test_decode_shape(self):
        img = np.random.default_rng(0).standard_normal((256, 256, 3))
        codes = main.vqvae_encode(img)
        # codebook random
        rng = np.random.default_rng(0)
        codebook = rng.standard_normal((8192, 16 * 16 * 3))
        out = main.vqvae_decode(codes, codebook, target_size=(256, 256, 3))
        self.assertEqual(out.shape, (256, 256, 3))


class TestVocab(unittest.TestCase):
    def test_size(self):
        v = main.chameleon_vocab()
        # 32000 + 8192 + 128 = 40320
        self.assertEqual(v, 40320)


class TestEncode(unittest.TestCase):
    def test_text(self):
        text = np.array([100, 200, 1000])
        out = main.encode_text(text, vocab_size_text=32000)
        np.testing.assert_array_equal(out, [100, 200, 1000])

    def test_text_out_of_vocab(self):
        text = np.array([100, 50000])  # 50000 > 32000
        out = main.encode_text(text, vocab_size_text=32000)
        self.assertEqual(out[0], 100)
        self.assertEqual(out[1], -1)

    def test_image_offset(self):
        codes = np.array([0, 100, 1000])
        out = main.encode_image_tokens(codes, vocab_size_text=32000, special_offset=64)
        # 0 + 32000 + 64 = 32064
        self.assertEqual(out[0], 32064)
        self.assertEqual(out[1], 32164)


class TestChameleonSequence(unittest.TestCase):
    def test_concat(self):
        text = np.array([10, 20, 30])
        image = np.array([0, 100, 200])
        seq = main.chameleon_sequence(text, image, vocab_size_text=32000)
        # 3 + 3 = 6
        self.assertEqual(seq.shape, (6,))


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