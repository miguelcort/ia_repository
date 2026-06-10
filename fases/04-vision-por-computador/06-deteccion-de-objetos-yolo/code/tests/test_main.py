"""Pruebas para 06-deteccion-de-objetos-yolo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestIoU(unittest.TestCase):
    def test_identidad(self):
        box = [0, 0, 10, 10]
        self.assertAlmostEqual(main.iou(box, box), 1.0)

    def test_sin_solapamiento(self):
        a = [0, 0, 10, 10]
        b = [20, 20, 30, 30]
        self.assertEqual(main.iou(a, b), 0.0)

    def test_parcial(self):
        a = [0, 0, 10, 10]
        b = [5, 5, 15, 15]
        # inter = 5*5 = 25, area_a=100, area_b=100, union=175
        self.assertAlmostEqual(main.iou(a, b), 25 / 175, places=3)


class TestNMS(unittest.TestCase):
    def test_nms_elimina_solapados(self):
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11]]
        scores = [0.9, 0.8]
        keep = main.nms(boxes, scores, iou_threshold=0.5)
        # Solo el primero sobrevive (mayor score)
        self.assertEqual(keep, [0])

    def test_nms_conservar_separados(self):
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        scores = [0.9, 0.7]
        keep = main.nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(set(keep), {0, 1})


class TestAnchors(unittest.TestCase):
    def test_anchors_count(self):
        a = main.generar_anchors()
        # 3 ratios x 3 scales = 9
        self.assertEqual(len(a), 9)

    def test_anchors_centrados(self):
        a = main.generar_anchors()
        # Cada anchor debe estar centrado en (0, 0)
        for anchor in a:
            cx = (anchor[0] + anchor[2]) / 2
            cy = (anchor[1] + anchor[3]) / 2
            self.assertAlmostEqual(cx, 0.0, places=5)
            self.assertAlmostEqual(cy, 0.0, places=5)


class TestXYConversion(unittest.TestCase):
    def test_conversion(self):
        box_xywh = [10, 20, 4, 6]
        box_xyxy = main.xywh_a_xyxy(box_xywh)
        self.assertEqual(box_xyxy, [8.0, 17.0, 12.0, 23.0])


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("NMS", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()