"""Pruebas para 21-puntos-clave-y-estimacion-de-poses."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestKeypoints(unittest.TestCase):
    def test_deteccion_count(self):
        kps = main.detectar_keypoints_mock(n_keypoints=17)
        self.assertEqual(len(kps), 17)

    def test_deteccion_rango(self):
        kps = main.detectar_keypoints_mock(H=64, W=64, n_keypoints=5)
        for x, y in kps:
            self.assertGreaterEqual(x, 10)
            self.assertLessEqual(x, 54)
            self.assertGreaterEqual(y, 10)
            self.assertLessEqual(y, 54)


class TestConexiones(unittest.TestCase):
    def test_conexiones_validas(self):
        kps = [(0, 0), (1, 1), (2, 2), (3, 3)]
        pares = main.conexiones_entre_keypoints(kps, [(0, 1), (1, 2), (5, 6)])
        # Solo (0,1) y (1,2) son validos; (5,6) no
        self.assertEqual(len(pares), 2)


class TestAngulo(unittest.TestCase):
    def test_angulo_recto(self):
        # 90 grados: a=(0,1), b=(0,0), c=(1,0)
        ang = main.angulo_articulacion((0, 1), (0, 0), (1, 0))
        self.assertAlmostEqual(ang, 90.0, places=3)

    def test_angulo_recto_negativo(self):
        ang = main.angulo_articulacion((1, 0), (0, 0), (0, 1))
        self.assertAlmostEqual(ang, 90.0, places=3)


class TestDistancia(unittest.TestCase):
    def test_distancia(self):
        d = main.distancia_entre_keypoints((0, 0), (3, 4))
        self.assertAlmostEqual(d, 5.0)


class TestPCK(unittest.TestCase):
    def test_pck_perfecto(self):
        pred = [(0, 0), (1, 1), (2, 2)]
        gt = [(0, 0), (1, 1), (2, 2)]
        self.assertEqual(main.pck_score(pred, gt, threshold=10), 1.0)

    def test_pck_fallo(self):
        pred = [(0, 0), (1, 1), (100, 100)]
        gt = [(0, 0), (1, 1), (2, 2)]
        # Solo 2 de 3 dentro de threshold
        self.assertAlmostEqual(main.pck_score(pred, gt, threshold=10), 2/3)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Keypoints", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()