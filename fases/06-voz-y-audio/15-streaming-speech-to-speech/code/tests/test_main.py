"""Pruebas para 15-streaming-speech-to-speech."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPipeline(unittest.TestCase):
    def test_streaming(self):
        audio = np.random.default_rng(0).normal(size=8000)  # 500ms
        texto = main.streaming_asr_to_llm_pipeline(audio)
        self.assertIsNotNone(texto)

    def test_chunk_corto(self):
        audio = np.random.default_rng(0).normal(size=1000)  # < 250ms
        self.assertIsNone(main.streaming_asr_to_llm_pipeline(audio))


class TestTTS(unittest.TestCase):
    def test_tts(self):
        tokens = main.tts_streaming_tokens("Hola mundo")
        self.assertGreater(len(tokens), 0)


class TestLatency(unittest.TestCase):
    def test_latency_natural(self):
        lat = main.latency_full_duplex()
        # 50 + 200 + 300 + 200 = 750ms
        self.assertEqual(lat, 750)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Latencia", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()