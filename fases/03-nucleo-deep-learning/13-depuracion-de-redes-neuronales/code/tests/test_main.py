"""Pruebas para 13-depuracion-de-redes-neuronales."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestGradientCheck(unittest.TestCase):
    def test_grad_correcto(self):
        f = lambda x: x ** 2
        df = lambda x: 2 * x
        diff = main.check_gradiente_numerico(f, df, np.array([3.0]))
        self.assertLess(diff, 1e-5)

    def test_grad_incorrecto(self):
        # Gradiente analitico incorrecto: dice 3x en vez de 2x
        f = lambda x: x ** 2
        df_incorrecto = lambda x: 3 * x
        diff = main.check_gradiente_numerico(f, df_incorrecto, np.array([3.0]))
        # 3x vs 2x en x=3: diferencia 3
        self.assertGreater(diff, 1.0)


class TestCheckNaNInf(unittest.TestCase):
    def test_sin_problemas(self):
        arr = np.array([1.0, 2.0, 3.0])
        self.assertIn("OK", main.check_nan_inf(arr))

    def test_nan(self):
        arr = np.array([1.0, np.nan, 3.0])
        self.assertIn("ERROR", main.check_nan_inf(arr))
        self.assertIn("NaN", main.check_nan_inf(arr))

    def test_inf(self):
        arr = np.array([1.0, np.inf, 3.0])
        self.assertIn("ERROR", main.check_nan_inf(arr))
        self.assertIn("Inf", main.check_nan_inf(arr))


class TestMonitorearPesos(unittest.TestCase):
    def test_pesos_normales(self):
        rng = np.random.default_rng(0)
        capas = [rng.normal(scale=0.1, size=(10, 5)) for _ in range(3)]
        reporte = main.monitorear_pesos(capas)
        # 3 lineas, una por capa
        self.assertEqual(len(reporte.split("\n")), 3)

    def test_pesos_sospechosos(self):
        # Pesos con std muy alta -> REVISAR
        capas = [np.random.default_rng(0).normal(scale=10.0, size=(10, 5))]
        reporte = main.monitorear_pesos(capas)
        self.assertIn("REVISAR", reporte)


class TestGradCheckPaso(unittest.TestCase):
    def test_ok(self):
        # f debe retornar escalar para que la diferencia finita tenga sentido
        f = lambda x: float(np.sum(x ** 2))
        df = lambda x: 2 * x
        resultado = main.grad_check_paso(f, df, np.array([1.0, 2.0, 3.0]))
        self.assertTrue(resultado["ok"])
        self.assertLess(resultado["diff_max"], 1e-3)

    def test_falla_con_nan(self):
        def f(x):
            return float(np.sum(x ** 2))
        def df_con_nan(x):
            return np.array([1.0, np.nan, 3.0])
        resultado = main.grad_check_paso(f, df_con_nan, np.array([1.0, 2.0, 3.0]))
        self.assertIn("NaN", resultado["nan_inf"])


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Gradient", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()