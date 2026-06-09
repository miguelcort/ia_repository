"""Pruebas para 12-operaciones-con-tensores."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestShape(unittest.TestCase):
    def test_escalar(self):
        self.assertEqual(main.shape(np.array(3.14)), ())

    def test_vector(self):
        self.assertEqual(main.shape(np.array([1, 2, 3])), (3,))

    def test_matriz(self):
        self.assertEqual(main.shape(np.zeros((2, 3))), (2, 3))

    def test_tensor_3d(self):
        self.assertEqual(main.shape(np.zeros((2, 3, 4))), (2, 3, 4))


class TestNdim(unittest.TestCase):
    def test_escalar(self):
        self.assertEqual(main.ndim(np.array(3.14)), 0)

    def test_vector(self):
        self.assertEqual(main.ndim(np.array([1, 2])), 1)

    def test_matriz(self):
        self.assertEqual(main.ndim(np.zeros((2, 3))), 2)

    def test_tensor_4d(self):
        self.assertEqual(main.ndim(np.zeros((2, 3, 4, 5))), 4)


class TestReshape(unittest.TestCase):
    def test_reshape_2d(self):
        t = np.arange(12)
        self.assertEqual(main.reshape(t, (3, 4)).shape, (3, 4))

    def test_reshape_total_elementos(self):
        t = np.arange(24)
        self.assertEqual(main.reshape(t, (2, 3, 4)).size, 24)


class TestTranspose(unittest.TestCase):
    def test_transpose_default(self):
        M = np.zeros((2, 3))
        self.assertEqual(main.transpose(M).shape, (3, 2))

    def test_transpose_con_axes(self):
        T = np.zeros((2, 3, 4))
        self.assertEqual(main.transpose(T, (0, 2, 1)).shape, (2, 4, 3))


class TestBroadcast(unittest.TestCase):
    def test_escalar_y_vector(self):
        a = np.array(3)
        b = np.array([1, 2, 3])
        np.testing.assert_array_equal(main.broadcast(a, b), [4, 5, 6])

    def test_vector_y_matriz(self):
        a = np.array([[1], [2]])
        b = np.array([[10, 20, 30]])
        resultado = main.broadcast(a, b)
        self.assertEqual(resultado.shape, (2, 3))


class TestMatmul(unittest.TestCase):
    def test_2d(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        np.testing.assert_array_equal(main.matmul(A, B), A @ B)

    def test_batch(self):
        rng = np.random.default_rng(0)
        A = rng.random((5, 3, 4))
        B = rng.random((5, 4, 2))
        self.assertEqual(main.matmul(A, B).shape, (5, 3, 2))


class TestEinsum(unittest.TestCase):
    def test_producto_punto(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(main.einsum("i,i->", a, b), 32)

    def test_matmul(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        np.testing.assert_array_equal(main.einsum("ij,jk->ik", A, B), A @ B)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Tensor 3D", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()