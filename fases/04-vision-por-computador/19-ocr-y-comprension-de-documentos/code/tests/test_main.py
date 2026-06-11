"""Pruebas para 19-ocr-y-comprension-de-documentos."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPreprocess(unittest.TestCase):
    def test_binarizar(self):
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        out = main.preprocesar_ocr(img, binario=True, umbral=128)
        # 0 -> 0, 128 -> 0, 255 -> 255
        np.testing.assert_array_equal(out, [[0, 0, 255]])


class TestProyecciones(unittest.TestCase):
    def test_proyeccion_h(self):
        img = np.full((10, 10), 255, dtype=np.uint8)
        img[2:4, :] = 0  # pixeles negros
        proj = main.proyectar_horizontal(img)
        # Filas 2 y 3 tienen 10 pixeles negros cada una (255 - 0 = 255)
        self.assertEqual(proj[2], 10 * 255)
        self.assertEqual(proj[3], 10 * 255)
        self.assertEqual(proj[0], 0)


class TestSegmentos(unittest.TestCase):
    def test_segmentos(self):
        proj = np.array([0, 0, 5, 5, 0, 0, 3, 3, 0])
        segs = main.segmentos_por_huecos(proj, min_gap=2)
        # Esperar 2 segmentos: [2, 4) y [6, 8)
        self.assertEqual(len(segs), 2)


class TestOCR(unittest.TestCase):
    def test_patch_vacio(self):
        patch = np.zeros((5, 5), dtype=np.uint8)
        self.assertEqual(main.simular_ocr_patch(patch), "")


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Lineas", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()