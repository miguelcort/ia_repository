"""Pruebas para 11-procesamiento-de-audio-en-tiempo-real."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVAD(unittest.TestCase):
    def test_habla(self):
        audio = np.random.default_rng(0).normal(scale=0.1, size=16000)
        self.assertEqual(main.vad_silero_mock(audio), 1)

    def test_silencio(self):
        audio = np.zeros(16000)
        self.assertEqual(main.vad_silero_mock(audio), 0)

    def test_vacio(self):
        self.assertEqual(main.vad_silero_mock(np.array([])), 0)


class TestStreaming(unittest.TestCase):
    def test_basico(self):
        sr = 16000
        audio = np.random.default_rng(0).normal(size=sr * 2)  # 2s
        chunks = main.streaming_asr_chunk(audio, sr, chunk_seconds=0.5)
        # 2s / 0.5s = 4 chunks
        self.assertEqual(len(chunks), 4)


class TestLatency(unittest.TestCase):
    def test_latency(self):
        lat = main.latency_streaming(model_latency_ms=200, chunk_ms=400)
        # 200 + 200 = 400
        self.assertEqual(lat, 400)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("VAD", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()