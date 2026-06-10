"""Pruebas para 04-funciones-de-activacion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSigmoid(unittest.TestCase):
    def test_cero(self):
        self.assertAlmostEqual(main.sigmoid(0.0), 0.5)

    def test_extremos(self):
        self.assertAlmostEqual(main.sigmoid(-100), 0.0, places=5)
        self.assertAlmostEqual(main.sigmoid(100), 1.0, places=5)

    def test_derivada_max(self):
        # Maximo de sigmoid' es 0.25 en z=0
        self.assertAlmostEqual(main.sigmoid_derivada(0.0), 0.25)


class TestTanh(unittest.TestCase):
    def test_cero(self):
        self.assertAlmostEqual(main.tanh(0.0), 0.0)

    def test_rango(self):
        z = np.array([-100.0, 100.0])
        np.testing.assert_array_almost_equal(main.tanh(z), [-1.0, 1.0])


class TestReLU(unittest.TestCase):
    def test_positivo(self):
        self.assertEqual(main.relu(2.0), 2.0)

    def test_negativo(self):
        self.assertEqual(main.relu(-2.0), 0.0)

    def test_cero(self):
        self.assertEqual(main.relu(0.0), 0.0)

    def test_derivada(self):
        z = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_equal(main.relu_derivada(z), [0.0, 0.0, 1.0])


class TestLeakyReLU(unittest.TestCase):
    def test_negativo(self):
        self.assertAlmostEqual(main.leaky_relu(-2.0), -0.02)

    def test_derivada_negativa(self):
        self.assertAlmostEqual(main.leaky_relu_derivada(-2.0), 0.01)


class TestGELU(unittest.TestCase):
    def test_cero(self):
        self.assertAlmostEqual(main.gelu(0.0), 0.0, places=5)


class TestSoftmax(unittest.TestCase):
    def test_suma_uno(self):
        logits = np.array([1.0, 2.0, 3.0])
        s = main.softmax(logits)
        self.assertAlmostEqual(s.sum(), 1.0)

    def test_estabilidad(self):
        # Numeros grandes no deben explotar
        logits = np.array([1000.0, 1001.0, 1002.0])
        s = main.softmax(logits)
        self.assertFalse(np.isnan(s).any())
        self.assertAlmostEqual(s.sum(), 1.0)

    def test_2d_batch(self):
        logits = np.array([[1.0, 2.0], [3.0, 4.0]])
        s = main.softmax(logits)
        # Suma 1 por fila
        self.assertTrue(np.allclose(s.sum(axis=-1), 1.0))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("softmax", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()