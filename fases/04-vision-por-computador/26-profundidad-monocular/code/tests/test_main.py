"""Pruebas para 26-profundidad-monocular."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDepth3D(unittest.TestCase):
    def test_depth_to_3d(self):
        depth = np.ones((10, 10)) * 5.0
        K = np.array([[100.0, 0, 5], [0, 100.0, 5], [0, 0, 1.0]])
        p3d = main.depth_to_3d(depth, K, (5, 5))
        # Centro de la imagen, depth=5, sin offset -> (0, 0, 5)
        self.assertAlmostEqual(p3d[0], 0.0, places=5)
        self.assertAlmostEqual(p3d[1], 0.0, places=5)
        self.assertAlmostEqual(p3d[2], 5.0)


class TestMetricas(unittest.TestCase):
    def test_delta_perfecto(self):
        depth = np.array([[1.0, 2.0], [3.0, 4.0]])
        delta = main.depth_evaluation(depth, depth)
        self.assertEqual(delta, 1.0)

    def test_rmse(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        # (0 + 1) / 2 = 0.5, sqrt = 0.707
        self.assertAlmostEqual(main.depth_rmse(a, b), np.sqrt(0.5))

    def test_mae(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        self.assertEqual(main.depth_mae(a, b), 0.5)

    def test_abs_rel(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 3.0])
        # (0/1 + 1/3) / 2 = 0.167
        self.assertAlmostEqual(main.absolute_relative_error(a, b), (0 + 1/3) / 2, places=3)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("delta", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()