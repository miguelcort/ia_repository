"""Pruebas para 05-arquitectura-whisper-y-fine-tuning."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestChunking(unittest.TestCase):
    def test_chunk_30s(self):
        sr = 16000
        senal = np.random.default_rng(0).normal(size=sr * 60)  # 1 min
        chunks = main.chunk_audio(senal, max_seconds=30, sample_rate=sr)
        self.assertEqual(len(chunks), 2)  # 60s / 30s = 2

    def test_chunk_corto(self):
        sr = 16000
        senal = np.random.default_rng(0).normal(size=sr * 5)  # 5s
        chunks = main.chunk_audio(senal, max_seconds=30, sample_rate=sr)
        self.assertEqual(len(chunks), 1)


class TestMelSpec(unittest.TestCase):
    def test_shape(self):
        senal = np.random.default_rng(0).normal(size=16000)  # 1s
        mel = main.mel_spectrogram_whisper(senal, n_mels=80, n_fft=400, hop=160)
        # 1s * 16000 = 16000 samples, frames = (16000-400)//160+1 = 98
        self.assertEqual(mel.shape[0], 80)


class TestTokenizar(unittest.TestCase):
    def test_tokeniza(self):
        tokens = main.tokenizar_texto_whisper("hola")
        self.assertGreater(len(tokens), 0)
        self.assertLess(len(tokens), 10)


class TestSpecialTokens(unittest.TestCase):
    def test_special(self):
        st = main.special_tokens_whisper()
        self.assertIn("<|startoftranscript|>", st)
        self.assertIn("<|es|>", st)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Chunks", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()