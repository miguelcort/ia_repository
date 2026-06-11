"""Pruebas para 19-tokenizacion-de-subpalabras."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestBPE(unittest.TestCase):
    def test_train_basico(self):
        corpus = "a b a a b c c c"
        merges, vocab = main.bpe_train(corpus, num_merges=5)
        # Debe haber generado al menos 1 merge
        self.assertGreater(len(merges), 0)
        self.assertGreater(len(vocab), 3)

    def test_apply(self):
        corpus = "el gato come"
        merges, _ = main.bpe_train(corpus, num_merges=5)
        tokens = main.bpe_apply("el gato", merges)
        # Los tokens deben reconstruir la palabra
        joined = "".join(t.replace("</w>", "") for t in tokens)
        self.assertIn("el", joined)
        self.assertIn("gato", joined)


class TestWordPiece(unittest.TestCase):
    def test_basico(self):
        vocab = {"el", "##gato", "gato", "come"}
        tokens = main.wordpiece_tokenizar("el gato", vocab)
        self.assertIn("el", tokens)
        self.assertIn("gato", tokens)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("BPE", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()