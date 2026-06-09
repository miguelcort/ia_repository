"""Pruebas para 01-que-es-machine-learning."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestGenerarDatos(unittest.TestCase):
    def test_forma(self):
        X, y = main.generar_datos_lineales(n=50)
        self.assertEqual(len(X), 50)
        self.assertEqual(len(y), 50)

    def test_reproducible(self):
        X1, y1 = main.generar_datos_lineales(semilla=0)
        X2, y2 = main.generar_datos_lineales(semilla=0)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_relacion_aproximada(self):
        X, y = main.generar_datos_lineales(n=1000, ruido=0.01)
        # y = 2x + 1 + ruido. Correlacion deberia ser muy alta
        correlacion = np.corrcoef(X, y)[0, 1]
        self.assertGreater(correlacion, 0.99)


class TestECM(unittest.TestCase):
    def test_perfecto_cero(self):
        y = np.array([1, 2, 3])
        self.assertEqual(main.error_cuadratico_medio(y, y), 0.0)

    def test_pequena_diferencia(self):
        y = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        ecm = main.error_cuadratico_medio(y, y_pred)
        self.assertAlmostEqual(ecm, 0.01)


class TestTrainTest(unittest.TestCase):
    def test_division(self):
        X = np.arange(100)
        y = X * 2
        X_tr, y_tr, X_te, y_te = main.dividir_train_test(X, y, prop_train=0.8)
        self.assertEqual(len(X_tr), 80)
        self.assertEqual(len(X_te), 20)
        # Sin solapamiento
        self.assertEqual(len(set(X_tr) & set(X_te)), 0)

    def test_reproducible(self):
        X = np.arange(50)
        y = X * 3
        _, _, X_te1, _ = main.dividir_train_test(X, y, semilla=42)
        _, _, X_te2, _ = main.dividir_train_test(X, y, semilla=42)
        np.testing.assert_array_equal(X_te1, X_te2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("ECM", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()