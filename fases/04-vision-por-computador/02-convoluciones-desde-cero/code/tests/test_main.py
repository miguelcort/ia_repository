"""Pruebas para 02-convoluciones-desde-cero."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestConv(unittest.TestCase):
    def test_identidad_simple(self):
        # Kernel identidad 3x3 con centro 1
        img = np.zeros((5, 5))
        img[2, 2] = 5.0
        kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
        out = main.conv2d(img, kernel, padding=1)
        # El pixel central debe dar 5
        self.assertEqual(out[2, 2], 5.0)

    def test_padding(self):
        img = np.ones((5, 5))
        kernel = np.ones((3, 3)) / 9.0
        # Sin padding: 3x3
        out = main.conv2d(img, kernel)
        self.assertEqual(out.shape, (3, 3))
        # Con padding=1: mismo tamano
        out_pad = main.conv2d(img, kernel, padding=1)
        self.assertEqual(out_pad.shape, (5, 5))

    def test_borde_vertical(self):
        # Imagen con borde vertical
        img = np.zeros((10, 10))
        img[:, 5:] = 1.0
        out = main.conv2d(img, main.kernel_borde_vertical())
        # Debe haber un maximo fuerte cerca del borde
        self.assertGreater(out.max(), 0.5)

    def test_blur(self):
        img = np.ones((5, 5))
        out = main.conv2d(img, main.kernel_blur())
        # Box blur de constante = constante
        np.testing.assert_array_almost_equal(out, np.ones((3, 3)))


class TestPool(unittest.TestCase):
    def test_max_pool(self):
        x = np.array([[1, 3], [2, 4]], dtype=float)
        out = main.max_pool2d(x, size=2, stride=2)
        # 2x2 maximo es 4
        self.assertEqual(out, 4.0)

    def test_max_pool_4x4(self):
        x = np.arange(16, dtype=float).reshape(4, 4)
        out = main.max_pool2d(x, size=2, stride=2)
        # 4 ventanas 2x2: max de cada una
        self.assertEqual(out.shape, (2, 2))
        self.assertEqual(out[0, 0], 5.0)  # [[0,1],[2,3]]
        self.assertEqual(out[0, 1], 7.0)  # [[4,5],[6,7]]

    def test_avg_pool(self):
        x = np.array([[1, 3], [2, 4]], dtype=float)
        out = main.avg_pool2d(x, size=2, stride=2)
        # promedio = 2.5
        self.assertAlmostEqual(out, 2.5)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Borde", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()