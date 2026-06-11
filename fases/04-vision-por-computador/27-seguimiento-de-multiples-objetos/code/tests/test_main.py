"""Pruebas para 27-seguimiento-de-multiples-objetos."""
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
        self.assertAlmostEqual(main.iou_box(box, box), 1.0)

    def test_sin_solapamiento(self):
        self.assertEqual(main.iou_box([0, 0, 5, 5], [10, 10, 15, 15]), 0.0)

    def test_matrix(self):
        a = [[0, 0, 10, 10], [20, 20, 30, 30]]
        b = [[0, 0, 10, 10], [20, 20, 30, 30]]
        mat = main.iou_matrix(a, b)
        self.assertEqual(mat.shape, (2, 2))
        # Diagonal deberia ser 1
        self.assertAlmostEqual(mat[0, 0], 1.0)
        self.assertAlmostEqual(mat[1, 1], 1.0)


class TestAssignment(unittest.TestCase):
    def test_simple(self):
        cost = np.array([[0.0, 1.0], [1.0, 0.1]])
        pares = main.hungarian_assignment(cost)
        # (0,0) y (1,1) por minimo
        self.assertEqual(len(pares), 2)


class TestKalman(unittest.TestCase):
    def test_step(self):
        x = np.array([0.0, 0.0, 1.0, 1.0])
        P = np.eye(4)
        x_new, P_new = main.kalman_step(x, P)
        self.assertEqual(x_new.shape, (4,))
        self.assertEqual(P_new.shape, (4, 4))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Kalman", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()