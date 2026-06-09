"""Pruebas para 20-transformada-fourier."""
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
    def test_fft_sin(self):
        t = np.linspace(0, 1, 128, endpoint=False)
        senal = np.sin(2 * np.pi * 5 * t)
        espectro = main.fft(senal)
        self.assertEqual(len(espectro), 128)

    def test_ifft_fft_identidad(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        recuperado = main.ifft(main.fft(x))
        np.testing.assert_allclose(recuperado.real, x, atol=1e-10)

    def test_fft_senhal_constante(self):
        x = np.ones(8)
        espectro = main.fft(x)
        # Pico en DC (frecuencia 0)
        self.assertGreater(abs(espectro[0]), 0)


class TestFrecuencias(unittest.TestCase):
    def test_fft_frecuencias_basica(self):
        freqs = main.fft_frecuencias(8, fs=1.0)
        # Debe tener 8 frecuencias, primera 0
        self.assertEqual(len(freqs), 8)
        self.assertEqual(freqs[0], 0.0)


class TestConvolucion(unittest.TestCase):
    def test_conv_fft_vs_numpy(self):
        x = np.array([1, 2, 3], dtype=float)
        y = np.array([0, 1, 0.5], dtype=float)
        resultado = main.conv_fft(x, y)
        esperado = np.convolve(x, y)
        np.testing.assert_allclose(resultado, esperado, atol=1e-10)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Magnitud", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()