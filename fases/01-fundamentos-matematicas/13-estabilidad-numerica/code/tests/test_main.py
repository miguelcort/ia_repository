"""Pruebas para 13-estabilidad-numerica."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEpsilon(unittest.TestCase):
    def test_epsilon_positivo(self):
        self.assertGreater(main.epsilon_maquina(), 0)

    def test_epsilon_cerca_2_neg_52(self):
        # Para float64, eps ~ 2.22e-16
        self.assertAlmostEqual(main.epsilon_maquina(), 2.22e-16, delta=1e-19)


class TestLimites(unittest.TestCase):
    def test_max_mayor_que_min(self):
        mx, mn = main.max_min()
        self.assertGreater(mx, mn)

    def test_max_positivo(self):
        mx, _ = main.max_min()
        self.assertGreater(mx, 0)


class TestCondicion(unittest.TestCase):
    def test_matriz_identidad(self):
        A = np.eye(3)
        self.assertAlmostEqual(main.condicion_matriz(A), 1.0)

    def test_matriz_casi_singular(self):
        A = np.array([[1.0, 1.0], [1.0, 1.0001]])
        self.assertGreater(main.condicion_matriz(A), 1000)


class TestSoftmax(unittest.TestCase):
    def test_suma_es_uno(self):
        x = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(main.softmax_estable(x).sum(), 1.0)

    def test_argumentos_grandes_estable(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        resultado = main.softmax_estable(x)
        self.assertAlmostEqual(resultado.sum(), 1.0)
        self.assertTrue(np.all(resultado > 0))

    def test_inestable_falla_con_grandes(self):
        # Softmax inestable produce NaN o Inf
        x = np.array([1000.0, 1001.0, 1002.0])
        resultado = main.softmax_inestable(x.copy())
        # Debe tener al menos un NaN o Inf
        self.assertTrue(np.any(np.isnan(resultado)) or np.any(np.isinf(resultado)) or np.all(resultado == 0))

    def test_estable_no_gradiente_explosivo(self):
        x = np.array([1.0, 2.0, 3.0, 1000.0])
        resultado = main.softmax_estable(x)
        # El mayor argumento debe tener la mayor probabilidad
        self.assertEqual(np.argmax(resultado), 3)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Epsilon", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()