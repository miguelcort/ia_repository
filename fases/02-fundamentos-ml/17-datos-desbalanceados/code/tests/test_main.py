"""Pruebas para 17-datos-desbalanceados."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestOversampling(unittest.TestCase):
    def test_balancea(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(size=(100, 2)), rng.normal(loc=[2, 2], size=(10, 2))])
        y = np.array([0] * 100 + [1] * 10)
        Xo, yo = main.oversampling_simple(X, y, ratio=1.0)
        # Ambas clases deberian tener 100
        _, counts = np.unique(yo, return_counts=True)
        self.assertEqual(counts[0], 100)
        self.assertEqual(counts[1], 100)

    def test_misma_clase_se_replica(self):
        # Caso con clases perfectamente balanceadas -> no se replica
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        y = np.array([0, 0, 1, 1])
        Xo, yo = main.oversampling_simple(X, y, ratio=1.0)
        # n_mayor=2, n_menor=2, target=2*1.0=2, replicas=0
        self.assertEqual(len(Xo), 4)


class TestUndersampling(unittest.TestCase):
    def test_balancea(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(size=(100, 2)), rng.normal(loc=[2, 2], size=(10, 2))])
        y = np.array([0] * 100 + [1] * 10)
        Xu, yu = main.undersampling_simple(X, y, ratio=1.0)
        _, counts = np.unique(yu, return_counts=True)
        # n_menor = 10, target = 10*1.0 = 10
        self.assertEqual(counts[0], 10)
        self.assertEqual(counts[1], 10)


class TestPesos(unittest.TestCase):
    def test_mayor_clase_menor_peso(self):
        y = np.array([0] * 100 + [1] * 10)
        pesos = main.pesos_por_clase(y)
        self.assertLess(pesos[y == 0].mean(), pesos[y == 1].mean())

    def test_suma_aprox_n(self):
        y = np.array([0] * 100 + [1] * 10)
        pesos = main.pesos_por_clase(y)
        # Por convencion, sum w_c = n
        self.assertAlmostEqual(pesos.sum(), len(y), places=5)


class TestConfusion(unittest.TestCase):
    def test_perfecto(self):
        y = np.array([0, 1, 0, 1])
        y_p = np.array([0, 1, 0, 1])
        cm = main.confusion_matriz(y, y_p, positiva=1)
        self.assertEqual(cm, {"tp": 2, "fp": 0, "fn": 0, "tn": 2})

    def test_todo_negativo(self):
        y = np.array([0, 1, 0, 1])
        y_p = np.array([0, 0, 0, 0])
        cm = main.confusion_matriz(y, y_p, positiva=1)
        self.assertEqual(cm["tp"], 0)
        self.assertEqual(cm["fn"], 2)
        self.assertEqual(cm["fp"], 0)
        self.assertEqual(cm["tn"], 2)


class TestF1(unittest.TestCase):
    def test_perfecto(self):
        cm = {"tp": 10, "fp": 0, "fn": 0, "tn": 90}
        self.assertEqual(main.f1_score(cm), 1.0)

    def test_cero(self):
        cm = {"tp": 0, "fp": 0, "fn": 10, "tn": 90}
        self.assertEqual(main.f1_score(cm), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Oversample", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()