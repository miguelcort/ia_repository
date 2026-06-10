"""Pruebas para 02-redes-multicapa."""
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
    def test_rango(self):
        z = np.array([-100.0, 0.0, 100.0])
        s = main.sigmoid(z)
        self.assertAlmostEqual(s[0], 0.0, places=5)
        self.assertAlmostEqual(s[1], 0.5)
        self.assertAlmostEqual(s[2], 1.0, places=5)

    def test_derivada(self):
        z = np.array([0.0])
        self.assertAlmostEqual(main.sigmoid_derivada(z)[0], 0.25)


class TestTanh(unittest.TestCase):
    def test_rango(self):
        z = np.array([0.0, 100.0, -100.0])
        t = main.tanh(z)
        self.assertAlmostEqual(t[0], 0.0)
        self.assertAlmostEqual(t[1], 1.0, places=5)
        self.assertAlmostEqual(t[2], -1.0, places=5)


class TestCapa(unittest.TestCase):
    def test_shape(self):
        capa = main.CapaDensa(3, 5)
        x = np.random.default_rng(0).normal(size=(4, 3))
        out = capa.forward(x)
        self.assertEqual(out.shape, (4, 5))

    def test_backward_shape(self):
        capa = main.CapaDensa(3, 5)
        x = np.random.default_rng(0).normal(size=(4, 3))
        capa.forward(x)
        grad_salida = np.ones((4, 5))
        grad_x = capa.backward(grad_salida)
        self.assertEqual(grad_x.shape, (4, 3))
        self.assertEqual(capa.grad_W.shape, (3, 5))
        self.assertEqual(capa.grad_b.shape, (5,))


class TestRed(unittest.TestCase):
    def test_forward(self):
        red = main.RedMulticapa([2, 3, 1], activacion="sigmoid")
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        out = red.forward(X)
        self.assertEqual(out.shape, (2, 1))

    def test_backward(self):
        red = main.RedMulticapa([2, 3, 1], activacion="sigmoid")
        X = np.array([[0.0, 0.0]])
        red.forward(X)
        grad = red.backward(np.ones((1, 1)))
        # Verifica que no hay NaN
        for capa in red.capas:
            self.assertFalse(np.isnan(capa.grad_W).any())


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("XOR", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()