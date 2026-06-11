"""Pruebas para 10-audio-transformers-whisper."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestLogMel(unittest.TestCase):
    def test_shape(self):
        audio = np.random.default_rng(0).standard_normal(16000)
        mel = main.log_mel_spectrogram(audio, n_mels=80, hop=160)
        # 16000 / 160 = 100 frames
        self.assertEqual(mel.shape, (100, 80))

    def test_log_aplicado(self):
        # Log-mel tiene rango [log(1e-6), log(energia_max + 1e-6)]
        audio = np.random.default_rng(0).standard_normal(16000)
        mel = main.log_mel_spectrogram(audio)
        # Cada valor > log(1e-6) = -13.8
        self.assertGreater(mel.min(), -14.0)

    def test_diferentes_n_mels(self):
        audio = np.random.default_rng(0).standard_normal(8000)
        mel_40 = main.log_mel_spectrogram(audio, n_mels=40)
        mel_128 = main.log_mel_spectrogram(audio, n_mels=128)
        self.assertEqual(mel_40.shape[1], 40)
        self.assertEqual(mel_128.shape[1], 128)


class TestConvEmbedding(unittest.TestCase):
    def test_downsample_2x(self):
        mel = np.random.default_rng(0).standard_normal((100, 80))
        W = np.random.default_rng(1).standard_normal((64, 80)) * 0.02
        b = np.zeros(64)
        out = main.conv_embedding(mel, W, b)
        self.assertEqual(out.shape[0], 50)
        self.assertEqual(out.shape[1], 64)

    def test_padding_si_impar(self):
        mel = np.random.default_rng(0).standard_normal((101, 80))
        W = np.random.default_rng(1).standard_normal((64, 80)) * 0.02
        b = np.zeros(64)
        out = main.conv_embedding(mel, W, b)
        # 101 -> pad a 102 -> 51
        self.assertEqual(out.shape[0], 51)


class TestWhisper(unittest.TestCase):
    def test_pipeline(self):
        audio = np.random.default_rng(0).standard_normal(16000)
        enc, dec = main.whisper_transcribe_mock(audio)
        # Encoder deberia tener T/2 frames
        self.assertLess(enc.shape[0], 100)
        # Decoder deberia tener text_len
        self.assertGreater(dec.shape[0], 0)


class TestDecode(unittest.TestCase):
    def test_decode(self):
        vocab = {"hola": 0, "mundo": 1, "<unk>": 2}
        out = main.decode_to_text([0, 1, 99], vocab)
        self.assertEqual(out, "hola mundo <unk>")


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