"""Pruebas para 01-fundamentos-de-audio."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSeno(unittest.TestCase):
    def test_shape(self):
        senal = main.generar_seno(440, 0.5, sample_rate=16000)
        # 0.5s * 16000 = 8000 samples
        self.assertEqual(len(senal), 8000)

    def test_amplitud(self):
        senal = main.generar_seno(440, 0.1, sample_rate=16000, amplitud=0.5)
        # Max abs debe ser 0.5
        self.assertLessEqual(float(np.abs(senal).max()), 0.5)


class TestUtilidades(unittest.TestCase):
    def test_sample_rate(self):
        self.assertTrue(main.sample_rate_valido(16000))
        self.assertTrue(main.sample_rate_valido(48000))
        self.assertFalse(main.sample_rate_valido(12345))

    def test_duracion(self):
        # 16000 samples a 16kHz = 1s
        self.assertAlmostEqual(main.duracion_samples(16000, 16000), 1.0)
        # 8000 samples a 16kHz = 0.5s
        self.assertAlmostEqual(main.duracion_samples(8000, 16000), 0.5)


class TestFeatures(unittest.TestCase):
    def test_energia(self):
        senal = np.ones(100)
        # RMS de senal constante = amplitud
        self.assertAlmostEqual(main.energia_rms(senal), 1.0)

    def test_zcr(self):
        senal = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        # Cada par de samples hay un cruce
        self.assertGreater(main.zero_crossing_rate(senal), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Senal", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()