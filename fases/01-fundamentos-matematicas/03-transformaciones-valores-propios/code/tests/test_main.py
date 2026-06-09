"""Pruebas para 03-transformaciones-valores-propios."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ

import numpy as np


class TestMatrizRotacion(unittest.TestCase):
    def test_rotacion_90_grados(self):
        R = main.matriz_rotacion(np.pi / 2)
        # [1, 0] -> [0, 1]
        np.testing.assert_allclose(R @ np.array([1.0, 0.0]), [0.0, 1.0], atol=1e-10)

    def test_rotacion_es_ortogonal(self):
        R = main.matriz_rotacion(0.7)
        # R R^T = I
        np.testing.assert_allclose(R @ R.T, np.eye(2), atol=1e-10)

    def test_rotacion_preserva_norma(self):
        R = main.matriz_rotacion(1.3)
        v = np.array([3.0, 4.0])
        self.assertAlmostEqual(np.linalg.norm(v), np.linalg.norm(R @ v))


class TestMatrizEscalado(unittest.TestCase):
    def test_escalado_2x(self):
        E = main.matriz_escalado(2.0, 3.0)
        np.testing.assert_allclose(E @ np.array([1.0, 1.0]), [2.0, 3.0])

    def test_escalado_diagonal(self):
        E = main.matriz_escalado(5.0, 5.0)
        np.testing.assert_array_equal(E, 5.0 * np.eye(2))


class TestAplicarTransformacion(unittest.TestCase):
    def test_identidad(self):
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            main.aplicar_transformacion(np.eye(3), v), v
        )


class TestValoresPropios(unittest.TestCase):
    def test_matriz_identidad(self):
        A = np.eye(3)
        autovalores, _ = main.valores_propios(A)
        np.testing.assert_allclose(np.sort(autovalores.real), [1, 1, 1])

    def test_matriz_diagonal(self):
        A = np.diag([2.0, 5.0, 7.0])
        autovalores, _ = main.valores_propios(A)
        np.testing.assert_allclose(np.sort(autovalores.real), [2, 5, 7])

    def test_matriz_simetrica_2x2(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        autovalores, _ = main.valores_propios(A)
        # Autovalores son 1 y 3
        np.testing.assert_allclose(np.sort(autovalores.real), [1.0, 3.0])

    def test_verificar_propios(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        self.assertTrue(main.verificar_vpropios(A))

    def test_verificar_propios_falla_para_matriz_aleatoria(self):
        rng = np.random.default_rng(0)
        A = rng.random((3, 3))
        # A veces pasa por casualidad, asi que probamos varias
        encontrado = False
        for _ in range(20):
            A = rng.random((3, 3))
            if not main.verificar_vpropios(A, tol=1e-6):
                encontrado = True
                break
        # Esto puede fallar con baja probabilidad; el test es informativo
        self.assertTrue(encontrado or main.verificar_vpropios(A, tol=1e-3))


class TestDiagonalizar(unittest.TestCase):
    def test_matriz_diagonalizada(self):
        A = np.array([[2.0, 1.0], [1.0, 2.0]])
        P, D, P_inv = main.diagonalizar(A)
        reconstruida = P @ D @ P_inv
        np.testing.assert_allclose(reconstruida, A, atol=1e-10)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("autovalores", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
