"""Pruebas para 13-vision-3d-nerf."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCamara(unittest.TestCase):
    def test_proyeccion_centro(self):
        # Punto en el eje optico (Z=1, X=Y=0): cae en (cx, cy)
        p = np.array([[0, 0, 1.0]])
        out = main.coordenadas_3d_a_2d(p, cx=320, cy=240)
        # out shape: (1, 2) con [u, v]
        u, v = out[0, 0], out[0, 1]
        self.assertAlmostEqual(u, 320.0)
        self.assertAlmostEqual(v, 240.0)

    def test_proyeccion_doble_distancia(self):
        # Doble distancia, mismo X/Y -> misma (u, v) porque X=Y=0
        p1 = np.array([[0, 0, 1.0]])
        p2 = np.array([[0, 0, 2.0]])
        u1 = main.coordenadas_3d_a_2d(p1)
        u2 = main.coordenadas_3d_a_2d(p2)
        # Ambas dan (cx, cy) porque X=Y=0
        self.assertAlmostEqual(u1[0, 0], u2[0, 0])


class TestRay(unittest.TestCase):
    def test_centro_pixel(self):
        # Pixel centro -> rayo en z=1
        d = main.rayo_direccion(np.array([320.0]), np.array([240.0]), cx=320, cy=240)
        np.testing.assert_array_almost_equal(d[0], [0, 0, 1], decimal=5)


class TestMuestreo(unittest.TestCase):
    def test_shape(self):
        origen = np.zeros(3)
        dir = np.array([[0, 0, 1]])
        puntos, t = main.muestrear_rayos(origen, dir, n_muestras=32)
        self.assertEqual(puntos.shape, (1, 32, 3))
        self.assertEqual(t.shape, (32,))


class TestComposicion(unittest.TestCase):
    def test_densidad_alta(self):
        # Densidad alta en todo el rayo -> pesos concentrados
        sigma = np.array([[1.0, 1.0, 1.0, 0.0]])
        color = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]])
        t = np.array([1.0, 2.0, 3.0, 4.0])
        delta = 1.0
        c_final, profundidad, _ = main.composicion_volumen(sigma, color, t, delta)
        # Profundidad deberia ser cercana a 1
        self.assertGreater(profundidad[0], 0)
        self.assertLess(profundidad[0], 4.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Color", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()