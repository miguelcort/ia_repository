"""Pruebas para 15-estadistica-para-ml."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEstadisticos(unittest.TestCase):
    def test_media(self):
        self.assertEqual(main.media(np.array([1, 2, 3, 4, 5])), 3.0)

    def test_mediana_impar(self):
        self.assertEqual(main.mediana(np.array([1, 2, 3, 4, 5])), 3.0)

    def test_mediana_par(self):
        self.assertEqual(main.mediana(np.array([1, 2, 3, 4])), 2.5)

    def test_varianza(self):
        # Varianza de [1,2,3,4,5] con ddof=0: 2.0
        self.assertAlmostEqual(main.varianza(np.array([1, 2, 3, 4, 5])), 2.0)

    def test_std(self):
        self.assertAlmostEqual(main.std(np.array([1, 2, 3, 4, 5])), np.sqrt(2.0))


class TestCorrelacion(unittest.TestCase):
    def test_pearson_perfecta(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(main.correlacion_pearson(x, x * 2 + 1), 1.0)

    def test_pearson_anticonica(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(main.correlacion_pearson(x, -x), -1.0)

    def test_spearman_monotona(self):
        x = np.array([1, 2, 3, 4, 5])
        # y monotono creciente
        y = np.array([10, 20, 30, 100, 200])
        self.assertAlmostEqual(main.spearman(x, y), 1.0)


class TestTTest(unittest.TestCase):
    def test_una_muestra_devuelve_dos_valores(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 100)
        resultado = main.t_test_una_muestra(x, mu=0.0)
        self.assertEqual(len(resultado), 2)

    def test_una_muestra_rechaza(self):
        rng = np.random.default_rng(0)
        x = rng.normal(5, 1, 100)
        t, p = main.t_test_una_muestra(x, mu=0.0)
        # Si scipy esta, p deberia ser muy pequeno
        if main.HAS_SCIPY:
            self.assertLess(p, 0.05)
        else:
            # Sin scipy, devuelve nan
            import math
            self.assertTrue(math.isnan(p))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Media", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()