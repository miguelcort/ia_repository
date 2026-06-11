"""Pruebas para 01-tokenizers."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCharLevel(unittest.TestCase):
    def test_basic(self):
        tokens = main.char_level_tokenize("hola")
        self.assertEqual(tokens, ['h', 'o', 'l', 'a'])


class TestWordLevel(unittest.TestCase):
    def test_basic(self):
        tokens = main.word_level_tokenize("Hola mundo, esto es un test.")
        self.assertIn("mundo", tokens)
        self.assertIn(",", tokens)


class TestBPE(unittest.TestCase):
    def test_pair_frequencies(self):
        vocab = {"h o l a": 2, "m u n d o": 1}
        pairs = main.compute_pair_frequencies(vocab)
        # Pair (h, o) deberia aparecer 2 veces
        self.assertEqual(pairs[("h", "o")], 2)

    def test_train_step(self):
        vocab = {"h o l a": 1, "h o l a m u n d o": 1}
        new_vocab, merges = main.bpe_train_step(vocab, num_merges=3)
        self.assertGreater(len(merges), 0)


class TestTokenizeWithVocab(unittest.TestCase):
    def test_basic(self):
        vocab = {"hola": 1, "mundo": 1}
        tokens = main.tokenize_with_vocab("holamundo", vocab)
        self.assertIn("hola", tokens)
        self.assertIn("mundo", tokens)


class TestSpecialTokens(unittest.TestCase):
    def test_specials(self):
        s = main.special_tokens_list()
        self.assertIn("BOS", s)
        self.assertIn("EOS", s)
        self.assertIn("PAD", s)


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