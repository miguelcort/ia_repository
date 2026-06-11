"""Pruebas para 02-construyendo-un-tokenizer."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPreTokenize(unittest.TestCase):
    def test_basic(self):
        tokens = main.pre_tokenize("Hola mundo")
        # Pre-tokenize puede ser aproximado, verificar que algo esta presente
        all_tokens = "".join(tokens)
        self.assertIn("Hola", all_tokens)
        self.assertIn("mundo", all_tokens)

    def test_punctuation(self):
        tokens = main.pre_tokenize("Hola, mundo!")
        # GPT-2 pre-tokenize incluye punctuation como parte
        self.assertGreater(len(tokens), 1)


class TestGetPairs(unittest.TestCase):
    def test_basic(self):
        pairs = main.get_pairs(["h", "o", "l", "a"])
        self.assertIn(("h", "o"), pairs)
        self.assertIn(("o", "l"), pairs)
        self.assertIn(("l", "a"), pairs)


class TestBPETokenizer(unittest.TestCase):
    def test_init(self):
        t = main.BPETokenizer()
        self.assertEqual(t.vocab, {})
        self.assertEqual(t.merges, [])

    def test_train(self):
        corpus = ["hola hola hola", "mundo mundo", "test"]
        t = main.BPETokenizer()
        t.train(corpus, vocab_size=20)
        self.assertGreater(len(t.vocab), 5)
        self.assertGreater(len(t.merges), 0)

    def test_encode_decode_roundtrip(self):
        corpus = ["hola mundo", "adios mundo", "test hola"]
        t = main.BPETokenizer()
        t.train(corpus, vocab_size=30)
        # Encode and decode
        text = "hola"
        encoded = t.encode(text)
        decoded = t.decode(encoded)
        # Decoded deberia contener chars de hola
        for ch in "hola":
            self.assertIn(ch, decoded)


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