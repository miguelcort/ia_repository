"""Pruebas para 11-generacion-de-audio."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMelSpectrogram(unittest.TestCase):
    def test_shape(self):
        audio = np.random.default_rng(0).standard_normal(22050)
        mel = main.mel_spectrogram_mock(audio, n_mels=80, hop=256)
        # 22050 / 256 = ~86 frames
        self.assertGreater(mel.shape[0], 80)
        self.assertEqual(mel.shape[1], 80)


class TestComponents(unittest.TestCase):
    def test_valle_cinco(self):
        comps = main.vall_e_components()
        self.assertEqual(len(comps), 5)

    def test_musicgen_cinco(self):
        comps = main.musicgen_components()
        self.assertEqual(len(comps), 5)

    def test_audio_models_seis(self):
        models = main.audio_models()
        self.assertEqual(len(models), 6)


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