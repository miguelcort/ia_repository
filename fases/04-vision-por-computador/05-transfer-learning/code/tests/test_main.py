"""Pruebas para 05-transfer-learning."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestExtraccion(unittest.TestCase):
    def test_features_shape(self):
        img = np.random.default_rng(0).normal(size=(32, 32, 3))
        feat = main.simulador_extraccion_features(img)
        # 32/4 = 8 bloques por dim, 8*8*3 = 192
        self.assertEqual(feat.shape, (1, 192))

    def test_cabeza(self):
        feat = np.random.default_rng(0).normal(size=(5, 100))
        logits = main.simulador_cabeza_clasificacion(feat, n_clases=10)
        self.assertEqual(logits.shape, (5, 10))


class TestLR(unittest.TestCase):
    def test_diferencial(self):
        lrs = main.estrategia_lr_diferencial(1e-3, 0.1)
        self.assertLess(lrs["backbone"], lrs["cabeza"])
        self.assertAlmostEqual(lrs["backbone"], 1e-4)


class TestCongelar(unittest.TestCase):
    def test_mascara(self):
        m = main.congelar_capas(10, 7)
        # 7 congeladas (False), 3 a entrenar (True)
        self.assertEqual(m.sum(), 3)
        self.assertEqual((~m).sum(), 7)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Features", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()