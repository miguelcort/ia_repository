"""Pruebas para 01-fundamentos-de-imagen."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCrearImagen(unittest.TestCase):
    def test_rayas_shape(self):
        img = main.crear_imagen_rayas(8, 8)
        self.assertEqual(img.shape, (8, 8))

    def test_rayas_alternadas(self):
        img = main.crear_imagen_rayas(4, 4, color=255)
        # Columnas pares son blancas
        self.assertEqual(img[0, 0], 255)
        self.assertEqual(img[0, 1], 0)
        self.assertEqual(img[0, 2], 255)

    def test_color_shape(self):
        img = main.crear_imagen_color(2, 3)
        self.assertEqual(img.shape, (2, 3, 3))
        # Canal R del pixel (0,0) = 255
        self.assertEqual(img[0, 0, 0], 255)


class TestRGBGrises(unittest.TestCase):
    def test_ya_gris(self):
        img = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        out = main.rgb_a_grises(img)
        self.assertEqual(out.shape, (2, 2))
        np.testing.assert_array_equal(out, img)

    def test_rgb_a_gris(self):
        img = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        out = main.rgb_a_grises(img)
        # R: 0.299 * 255 = 76
        # G: 0.587 * 255 = 150 (truncado a 149 en uint8)
        self.assertAlmostEqual(out[0, 0], 76, places=1)
        self.assertAlmostEqual(out[0, 1], 149, places=1)


class TestNormalizar(unittest.TestCase):
    def test_01(self):
        img = np.array([[0, 255]], dtype=np.uint8)
        out = main.normalizar_imagen(img, modo="01")
        self.assertAlmostEqual(out[0, 0], 0.0)
        self.assertAlmostEqual(out[0, 1], 1.0)

    def test_11(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        out = main.normalizar_imagen(img, modo="11")
        self.assertAlmostEqual(out[0, 0], -1.0)
        # 128/255 = 0.502 -> 0.502*2-1 = 0.004
        self.assertAlmostEqual(out[0, 1], 2 * 128 / 255 - 1, places=3)
        self.assertAlmostEqual(out[0, 2], 1.0)


class TestRedimensionar(unittest.TestCase):
    def test_mismo_tamano(self):
        img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        out = main.redimensionar_bilineal(img, 2, 2)
        # Debe ser identico o casi
        np.testing.assert_array_almost_equal(out, img)

    def test_subir(self):
        img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        out = main.redimensionar_bilineal(img, 4, 4)
        self.assertEqual(out.shape, (4, 4))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Imagen", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()