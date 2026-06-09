"""Pruebas para 08-optimizacion-familia-gradiente."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

sys.path[:] = RAIZ


class TestGD(unittest.TestCase):
    def test_encuentra_minimo_cuadratica(self):
        f = lambda x, y: (x - 3) ** 2 + (y + 2) ** 2
        w, hist = main.gd(f, [0.0, 0.0], max_iter=200)
        self.assertAlmostEqual(w[0], 3.0, places=1)
        self.assertAlmostEqual(w[1], -2.0, places=1)

    def test_perdida_decrece(self):
        f = lambda x: (x - 1) ** 2
        _, hist = main.gd(f, [0.0], max_iter=100)
        self.assertLess(hist[-1], hist[0])

    def test_convergencia_temprana(self):
        # Si el opt converge antes de max_iter, devuelve antes
        f = lambda x: (x - 5) ** 2
        _, hist = main.gd(f, [0.0], max_iter=10000, tol=1e-8)
        self.assertLess(len(hist), 1000)


class TestGDMomentum(unittest.TestCase):
    def test_minimo_cuadratica(self):
        f = lambda x, y: (x - 1) ** 2 + (y - 1) ** 2
        w, _ = main.gd_momentum(f, [0.0, 0.0], max_iter=200)
        self.assertAlmostEqual(w[0], 1.0, places=1)
        self.assertAlmostEqual(w[1], 1.0, places=1)


class TestAdam(unittest.TestCase):
    def test_minimo_cuadratica(self):
        f = lambda x, y: (x - 2) ** 2 + (y + 3) ** 2
        w, _ = main.adam(f, [0.0, 0.0], max_iter=2000, lr=0.1)
        self.assertAlmostEqual(w[0], 2.0, places=1)
        self.assertAlmostEqual(w[1], -3.0, places=1)

    def test_maneja_gradiente_cero(self):
        # Funcion constante: gradiente es 0, debe quedarse en w_inicial
        f = lambda x: 5.0
        w, _ = main.adam(f, [3.0], max_iter=50)
        self.assertAlmostEqual(w[0], 3.0, places=2)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("GD", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
