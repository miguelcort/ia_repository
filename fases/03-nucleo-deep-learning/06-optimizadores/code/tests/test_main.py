"""Pruebas para 06-optimizadores."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSGD(unittest.TestCase):
    def test_minimiza_x2(self):
        # f(x) = x^2, minimo en 0
        x, _ = main.optimizar(main.SGD(0.1), lambda x: x**2, lambda x: 2*x, np.array([5.0]), n_iter=100)
        self.assertLess(abs(x[0]), 0.1)

    def test_disminuye(self):
        # Verifica que el parametro se mueve en la direccion correcta
        opt = main.SGD(0.1)
        p = np.array([1.0])
        opt.actualizar([p], [np.array([2.0])])
        self.assertLess(p[0], 1.0)


class TestMomentum(unittest.TestCase):
    def test_disminuye(self):
        # Momentum con lr pequeno debe converger
        _, traj = main.optimizar(main.SGD_Momentum(0.01, 0.9), lambda x: x**2, lambda x: 2*x, np.array([5.0]), n_iter=100)
        self.assertLess(abs(traj[-1][0]), 1.0)


class TestAdaGrad(unittest.TestCase):
    def test_disminuye_lr(self):
        # AdaGrad con lr fijo debe decrecer el paso efectivo
        opt = main.AdaGrad(0.1)
        p = np.array([1.0])
        # Primer paso
        opt.actualizar([p], [np.array([1.0])])
        p1 = p[0]
        # Segundo paso con mismo gradiente: lr efectivo menor
        opt.actualizar([p], [np.array([1.0])])
        p2 = p[0]
        # El cambio |p1 - p2| debe ser menor que |p0 - p1| = 0.1
        self.assertLess(abs(p2 - p1), 0.1)


class TestRMSProp(unittest.TestCase):
    def test_minimiza(self):
        _, traj = main.optimizar(main.RMSProp(0.1), lambda x: x**2, lambda x: 2*x, np.array([3.0]), n_iter=100)
        self.assertLess(abs(traj[-1][0]), 0.5)


class TestAdam(unittest.TestCase):
    def test_minimiza(self):
        _, traj = main.optimizar(main.Adam(0.1), lambda x: x**2, lambda x: 2*x, np.array([5.0]), n_iter=100)
        self.assertLess(abs(traj[-1][0]), 0.1)

    def test_rapido(self):
        # Adam debe converger rapidamente con lr=0.5
        _, traj = main.optimizar(main.Adam(0.5), lambda x: x**2, lambda x: 2*x, np.array([10.0]), n_iter=50)
        self.assertLess(abs(traj[-1][0]), 0.5)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Adam", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()