"""Pruebas para 03-regresion-logistica."""
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
    def test_sigmoid_cero(self):
        self.assertAlmostEqual(main.sigmoid(0), 0.5)

    def test_sigmoid_positivo(self):
        self.assertGreater(main.sigmoid(2), 0.5)

    def test_sigmoid_negativo(self):
        self.assertLess(main.sigmoid(-2), 0.5)

    def test_sigmoid_extremo(self):
        self.assertAlmostEqual(main.sigmoid(1000), 1.0, places=5)
        self.assertAlmostEqual(main.sigmoid(-1000), 0.0, places=5)


class TestAjuste(unittest.TestCase):
    def test_datos_separables(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(-2, 0.5, (100, 2)), rng.normal(2, 0.5, (100, 2))])
        y = np.array([0] * 100 + [1] * 100)
        w = main.ajustar(X, y, lr=1.0, epochs=2000)
        y_pred = main.predecir(w, X)
        # Accuracy > 95% en datos separables
        self.assertGreater(main.accuracy(y, y_pred), 0.95)


class TestPrediccion(unittest.TestCase):
    def test_prediccion_binaria(self):
        w = np.array([0.0, 1.0, 0.0])  # intercepto=0, peso en x1
        X = np.array([[-1.0, 0.0], [1.0, 0.0]])
        y_pred = main.predecir(w, X, umbral=0.5)
        self.assertEqual(y_pred[0], 0)
        self.assertEqual(y_pred[1], 1)

    def test_umbral(self):
        w = np.array([0.0, 0.0])
        X = np.array([[0.0]])
        # Para todos ceros, proba = 0.5, justo en el umbral
        self.assertEqual(main.predecir(w, X, umbral=0.5)[0], 1)


class TestAccuracy(unittest.TestCase):
    def test_perfecta(self):
        y = np.array([0, 1, 0, 1])
        self.assertEqual(main.accuracy(y, y), 1.0)

    def test_cero(self):
        y = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        self.assertEqual(main.accuracy(y, y_pred), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Accuracy", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()