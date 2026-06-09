"""Pruebas para 16-metodos-de-muestreo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMuestreo(unittest.TestCase):
    def test_simple_n_50(self):
        poblacion = np.arange(1000)
        muestra = main.muestreo_aleatorio_simple(poblacion, 50)
        self.assertEqual(len(muestra), 50)
        # Todos los elementos son unicos (sin reemplazo)
        self.assertEqual(len(np.unique(muestra)), 50)

    def test_simple_reproducible(self):
        poblacion = np.arange(1000)
        m1 = main.muestreo_aleatorio_simple(poblacion, 50, semilla=0)
        m2 = main.muestreo_aleatorio_simple(poblacion, 50, semilla=0)
        np.testing.assert_array_equal(m1, m2)

    def test_con_reemplazo(self):
        poblacion = np.arange(10)
        muestra = main.muestreo_aleatorio_con_reemplazo(poblacion, 100, semilla=0)
        self.assertEqual(len(muestra), 100)
        # Puede haber repetidos


class TestEstratificado(unittest.TestCase):
    def test_estratificado(self):
        poblacion = np.arange(100)
        estratos = [np.arange(0, 33), np.arange(33, 66), np.arange(66, 100)]
        muestra = main.muestreo_estratificado(poblacion, estratos, [5, 5, 5])
        self.assertEqual(len(muestra), 15)


class TestBootstrap(unittest.TestCase):
    def test_devuelve_campos(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 50)
        res = main.bootstrap(x, np.mean, n_remuestras=100)
        self.assertIn("estimacion", res)
        self.assertIn("media_boot", res)
        self.assertIn("ic_95", res)
        self.assertEqual(len(res["ic_95"]), 2)

    def test_ic_contiene_media_verdadera(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5, 1, 200)
        res = main.bootstrap(x, np.mean, n_remuestras=500)
        # Para x ~ N(5, 1) con n=200, la media deberia estar en el IC
        self.assertGreater(res["ic_95"][1], 4.8)
        self.assertLess(res["ic_95"][0], 5.2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Bootstrap", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()