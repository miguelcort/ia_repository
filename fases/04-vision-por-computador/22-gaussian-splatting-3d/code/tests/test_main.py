"""Pruebas para 22-gaussian-splatting-3d."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestInicializar(unittest.TestCase):
    def test_init_gaussiana(self):
        g = main.inicializar_gaussiana([0, 0, 0], color=(1, 0, 0))
        self.assertEqual(g["pos"].shape, (3,))
        self.assertEqual(g["color"].shape, (3,))
        self.assertEqual(g["opacidad"], 1.0)


class TestProyeccion(unittest.TestCase):
    def test_proyeccion(self):
        g = main.inicializar_gaussiana([0, 0, 1])
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        out = main.proyectar_gaussiana_2d(g, P)
        self.assertIsNotNone(out)
        np.testing.assert_array_almost_equal(out["centro"], [0, 0])


class TestComposicion(unittest.TestCase):
    def test_composicion_vacia(self):
        color, alpha = main.composicion_gaussianas([], 10, 10)
        self.assertEqual(color.shape, (10, 10, 3))
        self.assertEqual(alpha.shape, (10, 10))
        # Sin gaussianas, alpha es 0
        self.assertEqual(alpha.sum(), 0.0)

    def test_composicion_un_pixel(self):
        gs = [{"centro": np.array([5.0, 5.0]), "color": np.array([1.0, 0.0, 0.0]), "opacidad": 1.0}]
        color, alpha = main.composicion_gaussianas(gs, 10, 10)
        self.assertEqual(alpha[5, 5], 1.0)
        np.testing.assert_array_almost_equal(color[5, 5], [1.0, 0.0, 0.0])


class TestDensidad(unittest.TestCase):
    def test_max_en_centro(self):
        g = main.inicializar_gaussiana([0, 0, 0], escala=1.0)
        d_centro = main.densidad_gaussiana(g, np.array([0, 0, 0]))
        d_lejos = main.densidad_gaussiana(g, np.array([5, 0, 0]))
        self.assertGreater(d_centro, d_lejos)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Densidad", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()