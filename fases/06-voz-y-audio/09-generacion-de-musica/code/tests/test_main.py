"""Pruebas para 09-generacion-de-musica."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPianoRoll(unittest.TestCase):
    def test_basico(self):
        audio = main.piano_roll_mock([(60, 0, 1.0)], duracion=2.0, sample_rate=16000)
        # 2s * 16000 = 32000
        self.assertEqual(len(audio), 32000)


class TestMelSpec(unittest.TestCase):
    def test_shape(self):
        audio = np.random.default_rng(0).normal(size=16000)
        mel = main.mel_spectrogram_audio(audio, n_mels=80, n_fft=400, hop=160)
        self.assertEqual(mel.shape[0], 80)


class TestMusicGen(unittest.TestCase):
    def test_genera(self):
        audio = main.musicgen_mock("rock", duracion=2, sample_rate=16000)
        self.assertEqual(len(audio), 32000)

    def test_determinista(self):
        a1 = main.musicgen_mock("test", duracion=1, sample_rate=16000)
        a2 = main.musicgen_mock("test", duracion=1, sample_rate=16000)
        np.testing.assert_array_equal(a1, a2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Audio", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()