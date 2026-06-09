"""Pruebas para 14-normas-y-distancias."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestNormas(unittest.TestCase):
    def test_l1(self):
        self.assertEqual(main.norma_l1(np.array([3, 4])), 7.0)

    def test_l2_3_4_5(self):
        self.assertAlmostEqual(main.norma_l2(np.array([3, 4])), 5.0)

    def test_lp(self):
        self.assertAlmostEqual(main.norma_lp(np.array([3, 4]), p=2), 5.0)
        self.assertEqual(main.norma_lp(np.array([3, 4]), p=1), 7.0)

    def test_linf(self):
        self.assertEqual(main.norma_linf(np.array([-7, 3, 5])), 7.0)

    def test_norma_vector_cero(self):
        self.assertEqual(main.norma_l2(np.zeros(5)), 0.0)


class TestDistancias(unittest.TestCase):
    def test_euclidiana(self):
        self.assertAlmostEqual(
            main.distancia_euclidiana(np.array([0, 0]), np.array([3, 4])), 5.0
        )

    def test_manhattan(self):
        self.assertEqual(
            main.distancia_manhattan(np.array([1, 2]), np.array([4, 6])), 7.0
        )

    def test_coseno_iguales(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(main.distancia_coseno(a, a), 0.0)

    def test_coseno_ortogonales(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        self.assertAlmostEqual(main.distancia_coseno(a, b), 1.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("d_euclidiana", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()