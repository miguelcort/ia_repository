"""Pruebas para 04-arboles-de-decision-random-forest."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEntropia(unittest.TestCase):
    def test_entropia_pura(self):
        self.assertEqual(main.entropia(np.array([0, 0, 0])), 0.0)

    def test_entropia_50_50(self):
        self.assertAlmostEqual(main.entropia(np.array([0, 1])), 1.0)

    def test_entropia_3_clases(self):
        self.assertAlmostEqual(main.entropia(np.array([0, 1, 2])), np.log2(3))


class TestGanancia(unittest.TestCase):
    def test_split_perfecto(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        # Split que separa perfectamente
        gan = main.ganancia_informacion(y, np.array([0, 1, 2]), np.array([3, 4, 5]))
        # Toda la entropia se va, ganancia = H(y)
        self.assertAlmostEqual(gan, 1.0)

    def test_split_inutil(self):
        # y = [0, 0, 0, 0, 0, 0] es puro: cualquier split da 0 de ganancia
        y = np.array([0, 0, 0, 0, 0, 0])
        gan = main.ganancia_informacion(y, np.array([0, 1, 2]), np.array([3, 4, 5]))
        self.assertAlmostEqual(gan, 0.0, places=5)


class TestMejorSplit(unittest.TestCase):
    def test_encuentra_feature_correcta(self):
        rng = np.random.default_rng(0)
        # Feature 0 separa las clases
        X = np.column_stack([
            np.concatenate([np.zeros(50), np.ones(50)]),  # separador
            rng.normal(size=100),  # ruido
        ])
        y = np.concatenate([np.zeros(50), np.ones(50)])
        f, _ = main.mejor_split(X, y)
        self.assertEqual(f, 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Mejor split", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()