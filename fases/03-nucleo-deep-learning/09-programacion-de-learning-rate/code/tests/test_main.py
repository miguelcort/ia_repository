"""Pruebas para 09-programacion-de-learning-rate."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestConstante(unittest.TestCase):
    def test_constante(self):
        for p in [0, 10, 100]:
            self.assertEqual(main.lr_constante(p, 0.1), 0.1)


class TestStep(unittest.TestCase):
    def test_step_decay(self):
        # cada=10, drop=0.1: en paso 0 -> 0.1, en paso 10 -> 0.01
        self.assertAlmostEqual(main.lr_step(0, 0.1, 0.1, 10), 0.1)
        self.assertAlmostEqual(main.lr_step(10, 0.1, 0.1, 10), 0.01)
        self.assertAlmostEqual(main.lr_step(20, 0.1, 0.1, 10), 0.001)


class TestExponential(unittest.TestCase):
    def test_decay(self):
        # gamma=0.5: en paso 0 -> 0.1, paso 1 -> 0.05
        self.assertAlmostEqual(main.lr_exponential(0, 0.1, 0.5), 0.1)
        self.assertAlmostEqual(main.lr_exponential(1, 0.1, 0.5), 0.05)
        self.assertAlmostEqual(main.lr_exponential(2, 0.1, 0.5), 0.025)


class TestCosine(unittest.TestCase):
    def test_inicio_max(self):
        # En paso 0, lr deberia ser lr_max
        self.assertAlmostEqual(main.lr_cosine(0, lr_max=0.1, lr_min=0.0, T_max=100), 0.1)

    def test_fin_min(self):
        # En paso T_max, lr deberia ser lr_min
        self.assertAlmostEqual(main.lr_cosine(100, lr_max=0.1, lr_min=0.0, T_max=100), 0.0, places=5)

    def test_monotono_decreciente(self):
        # El cosine deberia ser monotono decreciente de lr_max a lr_min
        lrs = [main.lr_cosine(p, 0.1, 0.0, 100) for p in range(0, 101, 10)]
        for i in range(len(lrs) - 1):
            self.assertGreaterEqual(lrs[i], lrs[i + 1])


class TestWarmupCosine(unittest.TestCase):
    def test_warmup_lineal(self):
        # En warmup, lr sube linealmente
        lr0 = main.lr_warmup_cosine(0, warmup=10, lr_max=0.1, T_max=100)
        lr5 = main.lr_warmup_cosine(5, warmup=10, lr_max=0.1, T_max=100)
        # paso 0: lr_max * 1/10 = 0.01
        self.assertAlmostEqual(lr0, 0.01)
        # paso 5: lr_max * 6/10 = 0.06
        self.assertAlmostEqual(lr5, 0.06)

    def test_warmup_max(self):
        # En paso warmup-1, deberia alcanzar lr_max
        lr = main.lr_warmup_cosine(9, warmup=10, lr_max=0.1, T_max=100)
        self.assertAlmostEqual(lr, 0.1)


class TestReduceOnPlateau(unittest.TestCase):
    def test_reduce_si_plateau(self):
        # Si las ultimas 3 no mejoran, reduce
        lr = main.lr_reduce_on_plateau([0.5, 0.4, 0.45, 0.46, 0.47], paciencia=3, factor=0.5)
        # ultimo 0.47 * 0.5 = 0.235
        self.assertAlmostEqual(lr, 0.235)

    def test_no_reduce_si_mejora(self):
        lr = main.lr_reduce_on_plateau([0.5, 0.4, 0.3, 0.2, 0.1], paciencia=3, factor=0.5)
        # Sigue mejorando, no reduce
        self.assertEqual(lr, 0.1)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("cosine", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()