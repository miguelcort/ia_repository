"""Pruebas para 13-codecs-de-audio-neurales."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEncodeDecode(unittest.TestCase):
    def test_shape_tokens(self):
        sr = 24000
        audio = np.random.default_rng(0).normal(size=sr)
        tokens, info = main.encodec_compress_mock(audio, n_codebooks=4, codebook_size=512)
        # 24000 / 320 = 75 frames
        self.assertEqual(tokens.shape[1], 75)
        self.assertEqual(tokens.shape[0], 4)

    def test_decode_shape(self):
        tokens = np.zeros((4, 100), dtype=int)
        decoded = main.decodec_decompress_mock(tokens)
        self.assertEqual(len(decoded), 32000)


class TestBitrate(unittest.TestCase):
    def test_bitrate(self):
        # 4 codebooks * 9 bits * 24000 / 320 / 1000 = 2.7 kbps
        br = main.bitrate_kbps(4, 512, 320, 24000)
        self.assertGreater(br, 0)

    def test_compression_ratio(self):
        self.assertEqual(main.compress_ratio(24000, 320), 75)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Bitrate", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()