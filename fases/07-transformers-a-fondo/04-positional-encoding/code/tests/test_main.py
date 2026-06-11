"""Pruebas para 04-positional-encoding."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSinusoidal(unittest.TestCase):
    def test_shape(self):
        pe = main.sinusoidal_pe(10, 16)
        self.assertEqual(pe.shape, (10, 16))

    def test_rango(self):
        # Sinusoidal siempre en [-1, 1]
        pe = main.sinusoidal_pe(100, 64)
        self.assertGreaterEqual(pe.min(), -1.0001)
        self.assertLessEqual(pe.max(), 1.0001)

    def test_primer_token_pe(self):
        # PE(0, *) = [sin(0), cos(0), sin(0), cos(0), ...]
        #         = [0, 1, 0, 1, ...]
        pe = main.sinusoidal_pe(1, 8)
        expected = np.array([[0, 1, 0, 1, 0, 1, 0, 1]], dtype=float)
        np.testing.assert_allclose(pe, expected, atol=1e-9)


class TestRoPE(unittest.TestCase):
    def test_preserva_shape(self):
        q = np.random.default_rng(0).standard_normal((5, 8))
        k = np.random.default_rng(1).standard_normal((5, 8))
        qr, kr = main.rope(q, k)
        self.assertEqual(qr.shape, q.shape)
        self.assertEqual(kr.shape, k.shape)

    def test_rotacion_preserva_norma(self):
        # Rotacion es isometria: ||q_rot|| = ||q||
        q = np.random.default_rng(2).standard_normal((4, 8))
        k = np.random.default_rng(3).standard_normal((4, 8))
        qr, kr = main.rope(q, k)
        np.testing.assert_allclose(np.linalg.norm(qr, axis=-1),
                                   np.linalg.norm(q, axis=-1), atol=1e-9)


class TestALiBi(unittest.TestCase):
    def test_shape(self):
        b = main.alibi_bias(8, 4)
        self.assertEqual(b.shape, (4, 8, 8))

    def test_diagonal_cero(self):
        # Pos 0 vs pos 0: distancia 0
        b = main.alibi_bias(8, 4)
        np.testing.assert_allclose(np.diagonal(b, axis1=-2, axis2=-1), 0.0, atol=1e-9)

    def test_slopes_decrecientes(self):
        # Slopes: 2^(-8/n * h) decrece con h
        b = main.alibi_bias(4, 4)
        # Cabezas 0 tienen slope mayor (mas negativo = menos bias en valor)
        # mean a traves de seq: cabeza 0 < cabeza 3 (mas negativo)
        means = b.mean(axis=(1, 2))
        self.assertLess(means[0], means[-1])


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