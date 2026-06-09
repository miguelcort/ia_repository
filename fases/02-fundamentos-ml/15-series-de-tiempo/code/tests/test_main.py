"""Pruebas para 15-series-de-tiempo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMediaMovil(unittest.TestCase):
    def test_ventana_constante(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ma = main.media_movil(x, ventana=3)
        # ma[2] = (1+2+3)/3 = 2
        self.assertAlmostEqual(ma[2], 2.0)
        # ma[4] = (3+4+5)/3 = 4
        self.assertAlmostEqual(ma[4], 4.0)

    def test_nan_al_inicio(self):
        x = np.array([1.0, 2.0, 3.0])
        ma = main.media_movil(x, ventana=2)
        self.assertTrue(np.isnan(ma[0]))


class TestDiferencias(unittest.TestCase):
    def test_orden_1(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        d = main.diferencias(x, orden=1)
        # 3-1=2, 6-3=3, 10-6=4
        np.testing.assert_array_almost_equal(d, [2.0, 3.0, 4.0])

    def test_orden_2(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        d = main.diferencias(x, orden=2)
        # diff2: 6-2*3+1=1, 10-2*6+3=1
        np.testing.assert_array_almost_equal(d, [1.0, 1.0])


class TestAutocorrelation(unittest.TestCase):
    def test_lag_0_es_uno(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        acf = main.autocorrelation(x, lag_max=3)
        self.assertAlmostEqual(acf[0], 1.0)

    def test_simetria(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=50)
        acf = main.autocorrelation(x, lag_max=5)
        # La autocorrelacion esta en [-1, 1]
        self.assertTrue(np.all(np.abs(acf) <= 1.0))


class TestARForecast(unittest.TestCase):
    def test_forecast_constante(self):
        x = np.ones(10) * 5.0
        forecast = main.ar_forecast(x, horizonte=3)
        # Serie constante -> forecast constante
        for v in forecast:
            self.assertAlmostEqual(v, 5.0, places=5)

    def test_forecast_longitud(self):
        x = np.linspace(0, 10, 50)
        forecast = main.ar_forecast(x, horizonte=5)
        self.assertEqual(len(forecast), 5)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Forecast", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()