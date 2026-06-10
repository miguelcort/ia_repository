"""Pruebas para 05-funciones-de-perdida."""
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

    def test_conocido(self):
        y = np.array([0.0, 0.0])
        y_p = np.array([1.0, 1.0])
        self.assertEqual(main.mse(y, y_p), 1.0)


class TestMAE(unittest.TestCase):
    def test_cero(self):
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(main.mae(y, y), 0.0)

    def test_constante(self):
        y = np.array([0.0, 0.0, 0.0])
        y_p = np.array([2.0, 2.0, 2.0])
        self.assertEqual(main.mae(y, y_p), 2.0)


class TestBCE(unittest.TestCase):
    def test_perfecto(self):
        y = np.array([1.0, 0.0])
        y_p = np.array([0.9999, 0.0001])
        self.assertLess(main.bce(y, y_p), 0.01)

    def test_clipping(self):
        # Predicciones en 0 o 1 no deberian explotar
        y = np.array([1.0, 0.0])
        y_p = np.array([1.0, 0.0])
        result = main.bce(y, y_p)
        self.assertFalse(np.isinf(result))
        self.assertFalse(np.isnan(result))


class TestCCE(unittest.TestCase):
    def test_suma_uno(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        # Verifica que las predicciones suman 1
        self.assertTrue(np.allclose(y_pred.sum(axis=1), 1.0))
        cce = main.cce(y_true, y_pred)
        self.assertGreater(cce, 0.0)

    def test_perfecto(self):
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.999, 0.0005, 0.0005]])
        self.assertLess(main.cce(y_true, y_pred), 0.01)


class TestHuber(unittest.TestCase):
    def test_error_pequeno(self):
        # Para |error| < delta, huber = 0.5 * error^2
        y = np.array([0.0])
        y_p = np.array([0.5])  # error = 0.5, < delta=1
        h = main.huber(y, y_p, delta=1.0)
        self.assertAlmostEqual(h, 0.125)

    def test_error_grande(self):
        # Para |error| > delta, huber = delta * |error| - 0.5 * delta^2
        y = np.array([0.0])
        y_p = np.array([2.0])  # error = 2, delta=1
        h = main.huber(y, y_p, delta=1.0)
        # 1*2 - 0.5*1 = 1.5
        self.assertAlmostEqual(h, 1.5)


class TestContraste(unittest.TestCase):
    def test_positivos_mas_altos(self):
        # logits_pos > logits_neg: loss 0
        loss = main.contraste(np.array([1.0]), np.array([0.0]), margen=0.5)
        self.assertEqual(loss, 0.0)

    def test_violacion(self):
        # logits_pos - logits_neg = 0, margen = 1.0: loss = 1.0
        loss = main.contraste(np.array([0.0]), np.array([0.0]), margen=1.0)
        self.assertAlmostEqual(loss, 1.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("MSE", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()