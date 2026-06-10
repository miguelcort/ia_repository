"""Pruebas para 01-el-perceptron."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPerceptron(unittest.TestCase):
    def test_aprende_and(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])
        p = main.Perceptron(n_features=2, lr=0.1, n_epocas=50)
        p.fit(X, y)
        self.assertEqual(p.score(X, y), 1.0)

    def test_aprende_or(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 1])
        p = main.Perceptron(n_features=2, lr=0.1, n_epocas=50)
        p.fit(X, y)
        self.assertEqual(p.score(X, y), 1.0)

    def test_no_aprende_xor(self):
        # XOR no es linealmente separable
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        p = main.Perceptron(n_features=2, lr=0.1, n_epocas=100)
        p.fit(X, y)
        # El perceptron no puede aprender XOR
        self.assertLess(p.score(X, y), 1.0)


class TestActivacion(unittest.TestCase):
    def test_step(self):
        z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        out = main.Perceptron(n_features=1).activar(z)
        np.testing.assert_array_equal(out, [0, 0, 1, 1, 1])


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("AND accuracy", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()