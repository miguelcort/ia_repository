"""Pruebas para 04-glove-y-fasttext."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCoocurrencia(unittest.TestCase):
    def test_shape(self):
        corpus = [["a", "b", "c"]]
        vocab = {"a": 0, "b": 1, "c": 2}
        M = main.coocurrencia(corpus, vocab, ventana=1)
        self.assertEqual(M.shape, (3, 3))
        # a-b co-ocurren
        self.assertGreater(M[0, 1], 0)
        # a no co-ocurre consigo misma
        self.assertEqual(M[0, 0], 0)


class TestFastText(unittest.TestCase):
    def test_subwords(self):
        subs = main.fasttext_subword("apple", n_min=3, n_max=5)
        # Debe incluir la palabra completa
        self.assertIn("<apple>", subs)
        # Subwords de tamano 3: app, ppl, ple
        self.assertIn("app", subs)
        self.assertIn("ppl", subs)
        self.assertIn("ple", subs)

    def test_subwords_corto(self):
        # Palabra mas corta que n_min -> solo la palabra completa
        subs = main.fasttext_subword("a", n_min=3, n_max=5)
        self.assertEqual(subs, ["<a>"])


class TestEmbedding(unittest.TestCase):
    def test_embedding_shape(self):
        subs = ["<test>", "tes", "est"]
        vocab_sub = {"<test>": 0, "tes": 1, "est": 2, "xyz": 3}
        emb = main.embedding_desde_subwords(subs, vocab_sub, dim=10)
        self.assertEqual(emb.shape, (10,))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Subwords", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()