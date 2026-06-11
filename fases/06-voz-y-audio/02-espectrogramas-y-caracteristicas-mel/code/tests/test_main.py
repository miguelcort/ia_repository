"""Pruebas para 02-espectrogramas-y-caracteristicas-mel."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestFFT(unittest.TestCase):
    def test_fft_impulso(self):
        # FFT de delta de Kronecker deberia ser 1 en todas las frecuencias
        senal = np.zeros(8)
        senal[0] = 1.0
        spec = np.abs(main.fft_simple(senal))
        for s in spec:
            self.assertAlmostEqual(float(s), 1.0, places=3)


class TestMelScale(unittest.TestCase):
    def test_hz_a_mel(self):
        # 1000 Hz -> ~1000 mel (formula O'Shaughnessy, casi igual)
        m = main.hz_a_mel(1000.0)
        self.assertAlmostEqual(m, 1000.0, places=-1)  # diferencia < 5

    def test_mel_a_hz_inversa(self):
        for hz in [0, 100, 500, 1000, 4000]:
            m = main.hz_a_mel(hz)
            hz_back = main.mel_a_hz(m)
            self.assertAlmostEqual(hz_back, hz, places=1)


class TestMelFilterbank(unittest.TestCase):
    def test_shape(self):
        fb = main.mel_filterbank(n_mels=80, n_fft=512, sample_rate=16000)
        self.assertEqual(fb.shape, (80, 257))

    def test_no_negativos(self):
        fb = main.mel_filterbank(n_mels=80, n_fft=512, sample_rate=16000)
        self.assertGreaterEqual(fb.min(), 0.0)


class TestMelSpec(unittest.TestCase):
    def test_shape(self):
        sr = 16000
        senal = np.random.default_rng(0).normal(size=sr)
        mel = main.mel_spectrogram(senal, sample_rate=sr, n_mels=80, n_fft=400, hop=160)
        # 1s * 16000 / 160 hop = 100 frames (aprox)
        self.assertEqual(mel.shape[0], 80)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Mel-spec", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()