"""Pruebas para 13-leyes-de-escalado."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestChinchillaLoss(unittest.TestCase):
    def test_loss_decrece_con_params(self):
        # Mas params -> menor loss
        l1 = main.chinchilla_loss(N=1e9, D=1e10)
        l2 = main.chinchilla_loss(N=10e9, D=1e10)
        self.assertLess(l2, l1)

    def test_loss_decrece_con_tokens(self):
        l1 = main.chinchilla_loss(N=1e9, D=1e10)
        l2 = main.chinchilla_loss(N=1e9, D=1e11)
        self.assertLess(l2, l1)

    def test_loss_positiva(self):
        l = main.chinchilla_loss(N=1e6, D=1e6)
        self.assertGreater(l, 0)

    def test_loss_asintota(self):
        # Loss converge a e cuando N y D son muy grandes
        l = main.chinchilla_loss(N=1e15, D=1e15)
        # e = 1.69, deberia estar cerca
        self.assertLess(l, 2.0)


class TestComputeOptimal(unittest.TestCase):
    def test_ratio_cerca_de_uno(self):
        # N y D deberian ser magnitudes similares
        C = 6 * 1e10 * 1e10
        N, D = main.compute_optimal(None, C)
        # Ratio entre 0.5 y 2
        self.assertGreater(N / D, 0.5)
        self.assertLess(N / D, 2.0)

    def test_N_D_positivos(self):
        N, D = main.compute_optimal(None, 1e20)
        self.assertGreater(N, 0)
        self.assertGreater(D, 0)


class TestScaleGPT(unittest.TestCase):
    def test_gpt3_compute(self):
        # GPT-3 175B * 300B tokens = ~3.15e23 FLOPs
        _, C = main.scale_gpt(n_params=175e9, n_tokens=300e9)
        # 6 * 175e9 * 300e9 = 3.15e23
        np.testing.assert_allclose(C, 3.15e23, rtol=1e-3)

    def test_loss_menor_con_mas_compute(self):
        l1, _ = main.scale_gpt(n_params=1e9, n_tokens=1e10)
        l2, _ = main.scale_gpt(n_params=10e9, n_tokens=1e11)
        self.assertLess(l2, l1)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()