"""Pruebas para 05-regla-de-la-cadena-y-autodiff."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


class TestDerivadaCadena(unittest.TestCase):
    def test_identidad(self):
        # composicion vacia = identidad
        self.assertEqual(main.derivada_cadena([], 5.0), 1.0)

    def test_una_funcion(self):
        # f(x) = x^2 -> f'(x) = 2x
        f = lambda x: x ** 2
        self.assertAlmostEqual(main.derivada_cadena([f], 3.0), 6.0, places=4)

    def test_dos_funciones(self):
        # f(x) = 2*x^2 + 1 -> f'(x) = 4x
        f1 = lambda x: x ** 2
        f2 = lambda x: 2 * x + 1
        self.assertAlmostEqual(main.derivada_cadena([f1, f2], 3.0), 12.0, places=4)

    def test_composicion_larga(self):
        # f(x) = sqrt(sin(x^2 + 1) + 1) -> validar con valor numerico
        import math
        f1 = lambda x: x ** 2 + 1
        f2 = lambda x: math.sin(x) + 1
        f3 = lambda x: math.sqrt(x)
        resultado = main.derivada_cadena([f1, f2, f3], 1.0)
        self.assertIsInstance(resultado, float)
        # Verificamos que da un numero razonable
        self.assertGreater(abs(resultado), 0)


class TestGradCheck(unittest.TestCase):
    def test_gradiente_de_cuadratica(self):
        f = lambda x, y: (x - 1) ** 2 + (y + 2) ** 2
        grad = main.grad_check(f, [1.0, -2.0])
        # En el minimo, el gradiente debe ser ~0
        self.assertAlmostEqual(grad[0], 0.0, places=3)
        self.assertAlmostEqual(grad[1], 0.0, places=3)

    def test_gradiente_fuera_del_minimo(self):
        f = lambda x, y: (x - 1) ** 2 + (y + 2) ** 2
        grad = main.grad_check(f, [0.0, 0.0])
        # En (0,0), gradiente = (2*(0-1), 2*(0+2)) = (-2, 4)
        self.assertAlmostEqual(grad[0], -2.0, places=3)
        self.assertAlmostEqual(grad[1], 4.0, places=3)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Composicion", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
