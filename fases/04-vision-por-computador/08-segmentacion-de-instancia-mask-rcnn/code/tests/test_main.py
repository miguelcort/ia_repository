"""Pruebas para 08-segmentacion-de-instancia-mask-rcnn."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMasksToBoxes(unittest.TestCase):
    def test_mascara_vacia(self):
        m = np.zeros((10, 10), dtype=bool)
        box = main.masks_to_boxes([m])
        self.assertEqual(box[0], [0, 0, 0, 0])

    def test_mascara_cuadrada(self):
        m = np.zeros((10, 10), dtype=bool)
        m[2:5, 3:7] = True
        box = main.masks_to_boxes([m])
        self.assertEqual(box[0], [3, 2, 7, 5])


class TestMaskIoU(unittest.TestCase):
    def test_identidad(self):
        m = np.zeros((10, 10), dtype=bool)
        m[2:5, 2:5] = True
        self.assertAlmostEqual(main.mask_iou(m, m), 1.0)

    def test_sin_solapamiento(self):
        a = np.zeros((10, 10), dtype=bool)
        a[0:3, 0:3] = True
        b = np.zeros((10, 10), dtype=bool)
        b[5:8, 5:8] = True
        self.assertEqual(main.mask_iou(a, b), 0.0)


class TestFormat(unittest.TestCase):
    def test_coco(self):
        m = np.zeros((5, 5), dtype=bool)
        m[0:2, 0:2] = True
        resultado = main.format_coco_segmentation([[0, 0, 2, 2]], [m], [1], [0.9])
        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado[0]["category_id"], 1)
        self.assertEqual(resultado[0]["score"], 0.9)
        self.assertEqual(resultado[0]["bbox"], [0, 0, 2, 2])


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Boxes", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()