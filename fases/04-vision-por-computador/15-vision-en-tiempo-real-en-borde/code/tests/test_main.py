"""Pruebas para 15-vision-en-tiempo-real-en-borde."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCuantizar(unittest.TestCase):
    def test_cuantizar_shape(self):
        w = np.random.default_rng(0).normal(size=(100,))
        q, scale, _ = main.cuantizar_int8(w)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(q.shape, (100,))
        self.assertGreater(scale, 0.0)

    def test_dequantizar_round_trip(self):
        w = np.array([[1.0, -2.0, 0.5], [0.0, 0.1, -0.1]])
        q, scale, _ = main.cuantizar_int8(w)
        w_back = main.dequantizar(q, scale)
        # Error debe ser menor a scale
        max_err = np.abs(w - w_back).max()
        self.assertLess(max_err, scale + 1e-3)


class TestConv(unittest.TestCase):
    def test_macs(self):
        m = main.macs_conv2d(32, 32, 3, 16, 3, 3, 32, 32)
        # 32*32*3*16*3*3 = 442368
        self.assertEqual(m, 32 * 32 * 3 * 16 * 3 * 3)

    def test_params(self):
        p = main.params_conv2d(3, 16, 3, 3)
        # 3*16*9 + 16 = 448
        self.assertEqual(p, 448)


class TestFPS(unittest.TestCase):
    def test_fps_jetson(self):
        fps = main.fps_yolo_nano("Jetson-Orin")
        self.assertGreater(fps, 30)

    def test_fps_cpu(self):
        fps = main.fps_yolo_nano("CPU-i7")
        self.assertLess(fps, 30)


class TestSize(unittest.TestCase):
    def test_size_mb(self):
        # 1M params a 32 bits = ~3.81 MB (1e6 * 4 / 1024^2)
        self.assertAlmostEqual(main.modelo_size_mb(1e6, 32), 1e6 * 32 / 8 / 1024 / 1024, places=2)
        # A 8 bits = ~0.95 MB
        self.assertAlmostEqual(main.modelo_size_mb(1e6, 8), 1e6 * 8 / 8 / 1024 / 1024, places=2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("YOLOv8", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()