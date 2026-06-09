"""Pruebas para 19-numeros-complejos."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import math


class TestOperaciones(unittest.TestCase):
    def test_suma(self):
        self.assertEqual(main.sumar((1, 2), (3, 4)), (4, 6))

    def test_multiplicar(self):
        # (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        self.assertEqual(main.multiplicar((1, 2), (3, 4)), (-5, 10))

    def test_i_al_cuadrado(self):
        # i*i = -1
        self.assertEqual(main.multiplicar((0, 1), (0, 1)), (-1, 0))

    def test_modulo(self):
        self.assertAlmostEqual(main.modulo((3, 4)), 5.0)
        self.assertAlmostEqual(main.modulo((0, 0)), 0.0)


class TestPolar(unittest.TestCase):
    def test_polar_a_rect(self):
        r, t = main.polar_a_rect(1, 0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(t, 0.0)

    def test_argumento(self):
        self.assertAlmostEqual(main.argumento((1, 0)), 0.0)
        self.assertAlmostEqual(main.argumento((0, 1)), math.pi / 2)


class TestFFT(unittest.TestCase):
    def test_fft_senal_constante(self):
        import numpy as np
        senal = np.ones(8)
        mag = main.fft_magnitud(senal)
        # Senal constante: pico en DC
        self.assertGreater(mag[0], 0)

    def test_fft_sin(self):
        import numpy as np
        t = np.linspace(0, 1, 64)
        senal = np.sin(2 * np.pi * 5 * t)
        mag = main.fft_magnitud(senal)
        # Pico cerca de la frecuencia 5
        self.assertGreater(mag[5], 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("a + b", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()