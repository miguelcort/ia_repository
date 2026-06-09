"""Pruebas para 02-vectores-matrices-operaciones."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ

import numpy as np


class TestProductoPunto(unittest.TestCase):
    def test_basico(self):
        self.assertEqual(main.producto_punto(np.array([1, 2, 3]), np.array([4, 5, 6])), 32)

    def test_vectores_diferentes(self):
        with self.assertRaises(ValueError):
            main.producto_punto(np.array([1, 2]), np.array([1, 2, 3]))

    def test_ortogonales_da_cero(self):
        a = np.array([1, 0])
        b = np.array([0, 1])
        self.assertEqual(main.producto_punto(a, b), 0)


class TestProductoExterior(unittest.TestCase):
    def test_forma_resultado(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5])
        resultado = main.producto_exterior(a, b)
        self.assertEqual(resultado.shape, (3, 2))

    def test_coincide_con_numpy(self):
        rng = np.random.default_rng(0)
        a = rng.random(4)
        b = rng.random(3)
        np.testing.assert_array_equal(main.producto_exterior(a, b), np.outer(a, b))


class TestProductoHadamard(unittest.TestCase):
    def test_elemento_a_elemento(self):
        a = np.array([2, 3, 4])
        b = np.array([5, 6, 7])
        np.testing.assert_array_equal(
            main.producto_hadamard(a, b), np.array([10, 18, 28])
        )


class TestNorma(unittest.TestCase):
    def test_vector_unitario(self):
        v = np.array([0, 0, 1])
        self.assertAlmostEqual(main.norma_l2(v), 1.0)

    def test_3_4_5(self):
        self.assertAlmostEqual(main.norma_l2(np.array([3.0, 4.0])), 5.0)


class TestCoseno(unittest.TestCase):
    def test_identicos_es_uno(self):
        a = np.array([1, 2, 3])
        self.assertAlmostEqual(main.coseno(a, a), 1.0, places=5)

    def test_ortogonales_es_cero(self):
        a = np.array([1, 0])
        b = np.array([0, 1])
        self.assertAlmostEqual(main.coseno(a, b), 0.0)

    def test_opuestos_es_menos_uno(self):
        a = np.array([1, 2, 3])
        b = -a
        self.assertAlmostEqual(main.coseno(a, b), -1.0, places=5)

    def test_vector_cero(self):
        self.assertEqual(main.coseno(np.array([0, 0]), np.array([1, 1])), 0.0)


class TestPropiedades(unittest.TestCase):
    def test_distributividad(self):
        self.assertTrue(main.verificar_propiedades()["distributividad"])

    def test_simetria_punto(self):
        self.assertTrue(main.verificar_propiedades()["simetria_punto"])

    def test_matmul_no_conmutativo(self):
        self.assertTrue(main.verificar_propiedades()["matmul_no_conmutativo"])

    def test_matmul_asociativo(self):
        self.assertTrue(main.verificar_propiedades()["matmul_asociativo"])


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("cos(a, b)", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
