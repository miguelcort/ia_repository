"""Pruebas para 06-probabilidad-y-distribuciones."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ

import numpy as np


class TestBernoulli(unittest.TestCase):
    def test_p01_p1(self):
        # P(X=1) = 0.7
        b = main.bernoulli(0.7)
        self.assertAlmostEqual(b(1), 0.7)
        self.assertAlmostEqual(b(0), 0.3)

    def test_otro_valor_es_cero(self):
        b = main.bernoulli(0.5)
        self.assertEqual(b(5), 0.0)


class TestBinomial(unittest.TestCase):
    def test_p0_y_p_n(self):
        b = binomial = main.binomial(10, 0.5)
        # P(X=0) = 0.5^10
        self.assertAlmostEqual(binomial(0), 0.5 ** 10, places=5)
        # P(X=10) = 0.5^10
        self.assertAlmostEqual(binomial(10), 0.5 ** 10, places=5)

    def test_valor_extremo_es_cero(self):
        b = main.binomial(10, 0.5)
        self.assertEqual(b(-1), 0.0)
        self.assertEqual(b(11), 0.0)

    def test_suma_pmf_es_uno(self):
        b = main.binomial(20, 0.3)
        total = sum(b(k) for k in range(21))
        self.assertAlmostEqual(total, 1.0, places=5)


class TestGaussiana(unittest.TestCase):
    def test_pdf_en_media(self):
        g = main.gaussiana(mu=0, sigma=1)
        from math import pi, sqrt
        esperado = 1 / sqrt(2 * pi)
        self.assertAlmostEqual(g(0), esperado, places=5)

    def test_pdf_es_positiva(self):
        g = main.gaussiana(mu=0, sigma=1)
        for x in [-3, -1, 0, 1, 3]:
            self.assertGreater(g(x), 0)

    def test_pdf_decae(self):
        g = main.gaussiana(mu=0, sigma=1)
        self.assertGreater(g(0), g(1))
        self.assertGreater(g(1), g(2))


class TestMuestreo(unittest.TestCase):
    def test_media_aproximada(self):
        muestras = main.muestrear_normal(5.0, 2.0, n=100000, semilla=0)
        # Media muestral debe estar cerca de 5
        self.assertAlmostEqual(main.media_muestral(muestras), 5.0, delta=0.1)

    def test_varianza_aproximada(self):
        muestras = main.muestrear_normal(5.0, 2.0, n=100000, semilla=0)
        # Varianza muestral debe estar cerca de 4
        self.assertAlmostEqual(main.varianza_muestral(muestras), 4.0, delta=0.2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Media muestral", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
