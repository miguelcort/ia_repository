"""Pruebas para 25-modelos-de-vision-lenguaje."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestTokens(unittest.TestCase):
    def test_image_tokens(self):
        img = np.random.default_rng(0).normal(size=(28, 28, 3))
        tokens = main.image_to_tokens(img, patch_size=14, dim=64)
        # 28/14 = 2, 2*2 = 4 patches
        self.assertEqual(tokens.shape, (4, 64))

    def test_text_tokens(self):
        tokens = main.text_to_tokens("hello", n_tokens=10, dim=64)
        self.assertEqual(tokens.shape, (10, 64))


class TestCompose(unittest.TestCase):
    def test_concat(self):
        img_tokens = np.zeros((4, 64))
        text_tokens = np.zeros((8, 64))
        composed = main.llava_compose(img_tokens, text_tokens)
        self.assertEqual(composed.shape, (12, 64))


class TestGreedy(unittest.TestCase):
    def test_decode(self):
        logits = np.array([0.1, 0.5, 0.3, 0.1])
        vocab = ["<eos>", "the", "cat", "dog"]
        out = main.greedy_decode(logits, vocab, max_new=3)
        self.assertEqual(out[0], "the")


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Composed", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()