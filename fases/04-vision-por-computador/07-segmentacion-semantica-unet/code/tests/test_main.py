"""Pruebas para 07-segmentacion-semantica-unet."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDownUp(unittest.TestCase):
    def test_downsample(self):
        x = np.arange(64, dtype=float).reshape(8, 8, 1)
        out = main.downsample(x, factor=2)
        self.assertEqual(out.shape, (4, 4, 1))

    def test_upsample(self):
        x = np.arange(16, dtype=float).reshape(4, 4, 1)
        out = main.upsample(x, factor=2)
        self.assertEqual(out.shape, (8, 8, 1))

    def test_up_down_no_identico(self):
        # Up-down con nearest no es identidad exacta (promedio de bloques)
        x = np.random.default_rng(0).normal(size=(8, 8, 1))
        down = main.downsample(x, 2)
        up = main.upsample(down, 2)
        # Verificamos que es el promedio en bloques
        self.assertEqual(up.shape, x.shape)


class TestDiceLoss(unittest.TestCase):
    def test_perfecto(self):
        y = np.array([[1, 1], [1, 1]], dtype=float)
        self.assertAlmostEqual(main.dice_loss(y, y), 0.0, places=5)

    def test_sin_solapamiento(self):
        y_true = np.array([[1, 0], [0, 0]], dtype=float)
        y_pred = np.array([[0, 0], [0, 1]], dtype=float)
        # inter=0, suma=2, dice = 2/(2) = 1
        self.assertGreater(main.dice_loss(y_true, y_pred), 0.9)


class TestIoU(unittest.TestCase):
    def test_perfecto(self):
        y = np.array([[1, 1], [1, 1]], dtype=float)
        self.assertAlmostEqual(main.iou_segmentation(y, y), 1.0, places=5)

    def test_parcial(self):
        y_true = np.array([[1, 1, 0], [1, 0, 0]], dtype=float)
        y_pred = np.array([[1, 0, 0], [1, 0, 0]], dtype=float)
        # inter=2, union=3 -> IoU = 0.667
        self.assertAlmostEqual(main.iou_segmentation(y_true, y_pred), 2/3, places=3)


class TestSkip(unittest.TestCase):
    def test_concat(self):
        enc = np.zeros((4, 4, 8))
        dec = np.zeros((4, 4, 16))
        out = main.unet_skip_connection(enc, dec)
        # 8 + 16 = 24 canales
        self.assertEqual(out.shape, (4, 4, 24))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Dice", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()