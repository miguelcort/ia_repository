"""Pruebas para 12-emu3-next-token-for-generation."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestTokenizer(unittest.TestCase):
    def test_size(self):
        # 32768 + 100352 + 128 = 133248
        v = main.emu3_tokenizer()
        self.assertEqual(v, 133248)


class TestImageTokens(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((256, 256, 3))
        tokens = main.emu3_image_tokens(img, vqvae=32768)
        self.assertEqual(tokens.shape, (256,))
        self.assertLess(tokens.max(), 32768)


class TestUnderstand(unittest.TestCase):
    def test_concat(self):
        img = np.array([10, 20, 30])
        text = np.array([100, 200])
        out = main.emu3_understand(img, text)
        # 3 + 2 = 5
        self.assertEqual(out.shape, (5,))

    def test_offsets(self):
        img = np.array([0, 1, 2])
        text = np.array([0, 1])
        out = main.emu3_understand(img, text)
        # image: 0, 1, 2 (no offset)
        # text: 0 + 32768 + 128 = 32896
        self.assertEqual(out[0], 0)
        self.assertEqual(out[2], 2)
        self.assertEqual(out[3], 32896)


class TestGenerate(unittest.TestCase):
    def test_length(self):
        # mock returns uniform -> sampled ids within vocab
        def mock_model(x):
            return np.zeros(main.emu3_tokenizer())
        gen = main.emu3_generate(np.array([1, 2, 3]), mock_model, n_new=8)
        # 3 + 8 = 11
        self.assertEqual(gen.shape, (11,))


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