"""Pruebas para 10-mini-framework."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestLinear(unittest.TestCase):
    def test_forward_shape(self):
        capa = main.Linear(3, 5)
        x = np.random.default_rng(0).normal(size=(4, 3))
        out = capa.forward(x)
        self.assertEqual(out.shape, (4, 5))

    def test_backward_shape(self):
        capa = main.Linear(3, 5)
        x = np.random.default_rng(0).normal(size=(4, 3))
        capa.forward(x)
        grad = np.ones((4, 5))
        grad_x = capa.backward(grad)
        self.assertEqual(grad_x.shape, (4, 3))
        self.assertEqual(capa.grad_W.shape, (3, 5))
        self.assertEqual(capa.grad_b.shape, (5,))


class TestActivaciones(unittest.TestCase):
    def test_relu_backward(self):
        r = main.ReLU()
        x = np.array([-1.0, 0.0, 1.0])
        r.forward(x)
        # x > 0 -> [F, F, T]
        grad = r.backward(np.array([1.0, 1.0, 1.0]))
        np.testing.assert_array_equal(grad, [0.0, 0.0, 1.0])

    def test_sigmoid_backward(self):
        s = main.Sigmoid()
        out = s.forward(np.array([0.0]))
        # sigmoid(0) = 0.5, derivada = 0.5 * 0.5 = 0.25
        grad = s.backward(np.array([1.0]))
        self.assertAlmostEqual(grad[0], 0.25)


class TestSequential(unittest.TestCase):
    def test_forward(self):
        modelo = main.Sequential(
            main.Linear(2, 4),
            main.ReLU(),
            main.Linear(4, 1),
        )
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        out = modelo.forward(x)
        self.assertEqual(out.shape, (2, 1))

    def test_backward(self):
        modelo = main.Sequential(
            main.Linear(2, 4),
            main.ReLU(),
            main.Linear(4, 1),
        )
        x = np.array([[0.0, 0.0]])
        modelo.forward(x)
        modelo.backward(np.ones((1, 1)))
        # Verifica que los gradientes no son NaN
        for _, _, g in modelo.parametros():
            self.assertFalse(np.isnan(g).any())

    def test_parametros(self):
        modelo = main.Sequential(
            main.Linear(2, 4),
            main.ReLU(),
            main.Linear(4, 1),
        )
        params = modelo.parametros()
        # 2 capas Linear -> 4 parametros (W, b por capa)
        self.assertEqual(len(params), 4)


class TestEntrenador(unittest.TestCase):
    def test_paso_disminuye_loss(self):
        # XOR deberia aprender con suficiente capacidad
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        modelo = main.Sequential(
            main.Linear(2, 8),
            main.ReLU(),
            main.Linear(8, 1),
            main.Sigmoid(),
        )
        trainer = main.Entrenador(modelo, main.AdamSimple(0.05), main.mse, main.mse_derivada)
        trainer.fit(X, y, epocas=300, verbose=False)
        # La loss deberia haber bajado
        self.assertLess(trainer.historial[-1], trainer.historial[0])


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