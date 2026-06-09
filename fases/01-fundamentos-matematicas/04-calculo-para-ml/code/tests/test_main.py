"""Pruebas para 04-calculo-para-ml."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


class TestDerivadaPolinomio(unittest.TestCase):
    def test_constante_da_vacio(self):
        # f(x) = 5 -> f'(x) = 0
        self.assertEqual(main.derivada_polinomio([5.0]), [])

    def test_lineal(self):
        # f(x) = 2x + 3 -> f'(x) = 2
        self.assertEqual(main.derivada_polinomio([3.0, 2.0]), [2.0])

    def test_cuadratica(self):
        # f(x) = 3x^2 + 2x + 1 -> f'(x) = 6x + 2
        self.assertEqual(main.derivada_polinomio([1.0, 2.0, 3.0]), [2.0, 6.0])

    def test_cubica(self):
        # f(x) = x^3 -> f'(x) = 3x^2
        self.assertEqual(main.derivada_polinomio([0.0, 0.0, 0.0, 1.0]), [0.0, 0.0, 3.0])


class TestEvaluarPolinomio(unittest.TestCase):
    def test_constante(self):
        self.assertEqual(main.evaluar_polinomio([5.0], 100.0), 5.0)

    def test_polinomio_2x_mas_3(self):
        self.assertEqual(main.evaluar_polinomio([3.0, 2.0], 4.0), 11.0)

    def test_cuadratica_en_cero(self):
        self.assertEqual(main.evaluar_polinomio([1.0, 2.0, 3.0], 0.0), 1.0)


class TestDerivadaNumerica(unittest.TestCase):
    def test_cuadratica_2(self):
        f = lambda x: 3 * x ** 2 + 2 * x + 1
        self.assertAlmostEqual(main.derivada_numerica(f, 2.0), 14.0, places=4)

    def test_seno_es_coseno(self):
        import math
        self.assertAlmostEqual(main.derivada_numerica(math.sin, 0.0), 1.0, places=4)


class TestGradiente(unittest.TestCase):
    def test_minimo_cuadratica_2d(self):
        g = lambda x, y: (x - 1) ** 2 + (y + 2) ** 2
        minimo = main.gradiente(g, [0.0, 0.0], max_iter=200, lr=0.2)
        self.assertAlmostEqual(minimo[0], 1.0, places=1)
        self.assertAlmostEqual(minimo[1], -2.0, places=1)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Minimo", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
