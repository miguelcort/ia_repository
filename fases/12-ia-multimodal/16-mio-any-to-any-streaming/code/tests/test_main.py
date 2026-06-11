"""Pruebas para 16-mio-any-to-any-streaming."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestTokenize(unittest.TestCase):
    def test_text(self):
        ids = main.mio_tokenize("text", "hello", vocab_size=1000)
        self.assertEqual(ids.shape, (5,))

    def test_image(self):
        img = np.random.default_rng(0).standard_normal((64, 64, 3))
        tokens = main.mio_tokenize("image", img)
        # 4*4 = 16
        self.assertEqual(tokens.shape, (16,))

    def test_audio(self):
        audio = np.random.default_rng(0).standard_normal(16000 * 2)  # 2s
        tokens = main.mio_tokenize("audio", audio)
        # 2*16000 / 320 = 100
        self.assertEqual(tokens.shape, (100,))

    def test_video(self):
        video = np.random.default_rng(0).standard_normal((8, 64, 64, 3))
        tokens = main.mio_tokenize("video", video)
        # 8 * 16 = 128
        self.assertEqual(tokens.shape, (128,))


class TestStreamingChunk(unittest.TestCase):
    def test_basic(self):
        tokens = np.arange(150)
        chunks = main.mio_streaming_chunk(tokens, chunk_size=64)
        # 64, 64, 22
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].shape, (64,))
        self.assertEqual(chunks[1].shape, (64,))
        self.assertEqual(chunks[2].shape, (22,))

    def test_empty(self):
        tokens = np.array([])
        chunks = main.mio_streaming_chunk(tokens, chunk_size=64)
        self.assertEqual(len(chunks), 0)


class TestStreamingDecode(unittest.TestCase):
    def test_basic(self):
        chunks = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        decoders = {"text": lambda x: f"text:{x.tolist()}"}
        result = main.mio_streaming_decode(chunks, decoders, "text")
        self.assertIn("text:", result)


class TestAnyToAny(unittest.TestCase):
    def test_text_to_text(self):
        out = main.mio_any_to_any("text", "hello", "text",
                                    encoder_fn=None, decoder_fn=lambda x, m: x)
        np.testing.assert_array_equal(out, [hash(c) % 133248 for c in "hello"])


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