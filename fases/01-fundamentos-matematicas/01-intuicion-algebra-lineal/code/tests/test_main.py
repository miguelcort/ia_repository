"""Pruebas para 01-intuicion-algebra-lineal."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ

import numpy as np


class TestSumar(unittest.TestCase):
    def test_suma_basica(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        np.testing.assert_array_equal(
            main.sumar(A, B), np.array([[6, 8], [10, 12]])
        )

    def test_formas_incompatibles(self):
        A = np.zeros((2, 3))
        B = np.zeros((3, 2))
        with self.assertRaises(ValueError):
            main.sumar(A, B)


class TestEscalar(unittest.TestCase):
    def test_multiplicacion_por_escalar(self):
        A = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(
            main.escalar(2.0, A), np.array([[2, 4], [6, 8]])
        )

    def test_escalar_cero(self):
        A = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(main.escalar(0, A), np.zeros_like(A))


class TestTransponer(unittest.TestCase):
    def test_transpuesta_2x3(self):
        A = np.array([[1, 2, 3], [4, 5, 6]])
        esperado = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(main.transponer(A), esperado)

    def test_transpuesta_doble_es_identidad(self):
        A = np.random.default_rng(0).random((3, 3))
        np.testing.assert_array_equal(main.transponer(main.transponer(A)), A)


class TestMultiplicar(unittest.TestCase):
    def test_multiplicar_2x2(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        resultado = main.multiplicar(A, B)
        esperado = A @ B
        np.testing.assert_allclose(resultado, esperado)

    def test_coincide_con_numpy(self):
        rng = np.random.default_rng(42)
        A = rng.random((3, 4))
        B = rng.random((4, 2))
        manual = main.multiplicar(A, B)
        numpy_result = A @ B
        np.testing.assert_allclose(manual, numpy_result, atol=1e-10)

    def test_dimensiones_incompatibles(self):
        A = np.zeros((2, 3))
        B = np.zeros((2, 2))
        with self.assertRaises(ValueError):
            main.multiplicar(A, B)

    def test_requiere_2d(self):
        with self.assertRaises(ValueError):
            main.multiplicar(np.array([1, 2, 3]), np.array([4, 5, 6]))


class TestNorma(unittest.TestCase):
    def test_vector_3_4_da_5(self):
        v = np.array([3.0, 4.0])
        self.assertAlmostEqual(main.norma(v), 5.0)

    def test_vector_cero(self):
        v = np.zeros(5)
        self.assertAlmostEqual(main.norma(v), 0.0)

    def test_unidad(self):
        v = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(main.norma(v), 1.0)


class TestProductoPunto(unittest.TestCase):
    def test_producto_basico(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(main.producto_punto(a, b), 32)

    def test_coincide_con_numpy(self):
        rng = np.random.default_rng(0)
        a = rng.random(5)
        b = rng.random(5)
        self.assertAlmostEqual(main.producto_punto(a, b), float(np.dot(a, b)))

    def test_vectores_diferentes_falla(self):
        with self.assertRaises(ValueError):
            main.producto_punto(np.array([1, 2]), np.array([1, 2, 3]))


class TestMain(unittest.TestCase):
    def test_main_ejecuta_sin_error(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("A + B", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
