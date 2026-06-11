"""Pruebas para 10-modelos-de-lenguaje-de-audio."""
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
    def test_encode_shape(self):
        audio = np.random.default_rng(0).normal(size=24000)
        tokens = main.encode_audio_tokens(audio, n_codebooks=4, codebook_size=512)
        # 24000 / 320 = 75 frames
        self.assertEqual(tokens.shape[1], 75)
        self.assertEqual(tokens.shape[0], 4)

    def test_decode_shape(self):
        tokens = np.zeros((4, 100), dtype=int)
        decoded = main.decode_audio_tokens(tokens)
        # 100 * 320 = 32000
        self.assertEqual(len(decoded), 32000)


class TestEmbeddings(unittest.TestCase):
    def test_shape(self):
        audio = np.random.default_rng(0).normal(size=16000)
        emb = main.multimodal_embedding_audio(audio, dim=512)
        self.assertEqual(emb.shape, (512,))

    def test_text_shape(self):
        emb = main.text_embedding("test", dim=512)
        self.assertEqual(emb.shape, (512,))


class TestClap(unittest.TestCase):
    def test_identidad(self):
        v = np.array([1.0, 0.0])
        self.assertAlmostEqual(main.clap_similarity(v, v), 1.0)

    def test_ortogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(main.clap_similarity(a, b), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Audio tokens", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()