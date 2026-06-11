"""Pruebas para 19-audio-language-whisper-to-af3."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMelFeatures(unittest.TestCase):
    def test_shape(self):
        audio = np.random.default_rng(0).standard_normal(16000)
        feats = main.mel_spectrogram_features(audio, n_mels=80, hop=160)
        # 16000 / 160 = 100
        self.assertEqual(feats.shape, (100, 80))


class TestWhisperEncoder(unittest.TestCase):
    def test_shape(self):
        audio = np.random.default_rng(0).standard_normal(16000)
        enc = main.whisper_encoder(audio, d_model=512)
        self.assertEqual(enc.shape, (100, 512))


class TestWhisperDecoder(unittest.TestCase):
    def test_shape(self):
        audio = np.random.default_rng(0).standard_normal(16000)
        ctx = main.whisper_encoder(audio, d_model=512)
        prefix = np.array([1, 2, 3])
        out = main.whisper_decoder(prefix, ctx, vocab_size=1000, d_model=512)
        # 3 prefix + 100 audio = 103
        self.assertEqual(out.shape, (103, 1000))


class TestAudioTextSim(unittest.TestCase):
    def test_self(self):
        a = np.array([1.0, 0.0])
        sim = main.audio_text_similarity(a, a)
        self.assertAlmostEqual(sim, 1.0, places=8)


class TestAudioLM(unittest.TestCase):
    def test_concat(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((5, 8))
        t = rng.standard_normal((3, 8))
        out = main.audio_lm_forward(a, t)
        self.assertEqual(out.shape, (8, 8))


class TestAF3Unified(unittest.TestCase):
    def test_audio_only(self):
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(16000)
        out = main.af3_unified(audio=audio)
        self.assertEqual(out.shape[1], 512)

    def test_text_only(self):
        out = main.af3_unified(text="hello world")
        self.assertEqual(out.shape, (2, 512))

    def test_multimodal(self):
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(16000)
        img = rng.standard_normal((64, 64, 3))
        out = main.af3_unified(audio=audio, text="hello", image=img)
        # 100 audio + 1 text + 17 image = 118
        self.assertEqual(out.shape[0], 100 + 1 + 17)

    def test_empty(self):
        out = main.af3_unified()
        self.assertIsNone(out)


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