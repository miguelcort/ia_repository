"""Pruebas para 07-texto-a-habla-tts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPhonemes(unittest.TestCase):
    def test_basico(self):
        phs = main.text_to_phonemes_mock("Hola 123")
        self.assertIn("h", phs)
        self.assertNotIn("1", phs)  # Solo letras


class TestMel(unittest.TestCase):
    def test_shape(self):
        mel = main.mel_spectrogram_tts("test", n_mels=80, n_frames=200)
        self.assertEqual(mel.shape, (80, 200))


class TestVocoder(unittest.TestCase):
    def test_shape(self):
        mel = main.mel_spectrogram_tts("test", n_mels=80, n_frames=100)
        audio = main.vocoder_mock(mel)
        # 100 frames * 256 hop = 25600
        self.assertEqual(audio.shape, (25600,))


class TestPitch(unittest.TestCase):
    def test_pitch_positivo(self):
        sr = 22050
        # Senal con freq 220 Hz
        t = np.arange(sr // 4) / sr
        senal = np.sin(2 * np.pi * 220 * t)
        pitch = main.pitch_extraction_mock(senal, sample_rate=sr)
        self.assertGreater(pitch, 100)
        self.assertLess(pitch, 400)

    def test_pitch_silencio(self):
        senal = np.zeros(1000)
        self.assertEqual(main.pitch_extraction_mock(senal, sample_rate=22050), 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Fonemas", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()