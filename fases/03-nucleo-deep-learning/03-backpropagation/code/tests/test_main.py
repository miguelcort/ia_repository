"""Pruebas para 03-backpropagation."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMSE(unittest.TestCase):
    def test_cero(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(main.mse(y, y), 0.0)

    def test_positivo(self):
        y = np.array([0.0])
        y_p = np.array([1.0])
        self.assertEqual(main.mse(y, y_p), 1.0)


class TestRed(unittest.TestCase):
    def test_forward_shape(self):
        red = main.RedBP([2, 4, 1], lr=0.1)
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        out = red.forward(X)
        self.assertEqual(out.shape, (2, 1))

    def test_backward_grads(self):
        red = main.RedBP([2, 3, 1], lr=0.1)
        X = np.array([[0.0, 0.0]])
        y = np.array([[1.0]])
        y_pred = red.forward(X)
        red.backward(y, y_pred)
        # grad_W y grad_b deben existir y tener la forma correcta
        for capa in red.capas:
            self.assertIn("grad_W", capa)
            self.assertIn("grad_b", capa)
            self.assertFalse(np.isnan(capa["grad_W"]).any())

    def test_aprende_xor(self):
        # Backprop completo debe aprender XOR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        red = main.RedBP([2, 8, 1], lr=1.0)
        red.fit(X, y, epocas=2000, verbose=False)
        y_pred = red.forward(X)
        # Threshold 0.5
        pred_bin = (y_pred > 0.5).astype(int)
        self.assertEqual(np.mean(pred_bin == y), 1.0)


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