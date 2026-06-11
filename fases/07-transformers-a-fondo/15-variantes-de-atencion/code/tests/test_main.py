"""Pruebas para 15-variantes-de-atencion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestStandardAttention(unittest.TestCase):
    def test_shape(self):
        Q = np.random.default_rng(0).standard_normal((4, 8))
        K = np.random.default_rng(1).standard_normal((4, 8))
        V = np.random.default_rng(2).standard_normal((4, 8))
        out, w = main.standard_attention(Q, K, V)
        self.assertEqual(out.shape, (4, 8))
        self.assertEqual(w.shape, (4, 4))


class TestLinearAttention(unittest.TestCase):
    def test_shape(self):
        Q = np.random.default_rng(0).standard_normal((4, 8))
        K = np.random.default_rng(1).standard_normal((4, 8))
        V = np.random.default_rng(2).standard_normal((4, 8))
        out, _ = main.linear_attention(Q, K, V)
        self.assertEqual(out.shape, (4, 8))

    def test_lineal_es_diferente_a_standard(self):
        # Linear attention aproxima standard pero no es identica
        Q = np.random.default_rng(0).standard_normal((4, 8))
        K = np.random.default_rng(1).standard_normal((4, 8))
        V = np.random.default_rng(2).standard_normal((4, 8))
        out_s, _ = main.standard_attention(Q, K, V)
        out_l, _ = main.linear_attention(Q, K, V, kernel="elu")
        # No es identica pero no infinita
        self.assertFalse(np.allclose(out_s, out_l, atol=1e-3))
        self.assertTrue(np.isfinite(out_l).all())


class TestSlidingWindow(unittest.TestCase):
    def test_attended_range(self):
        n = 8
        Q = np.random.default_rng(0).standard_normal((n, 4))
        K = np.random.default_rng(1).standard_normal((n, 4))
        V = np.random.default_rng(2).standard_normal((n, 4))
        _, w = main.sliding_window_attention(Q, K, V, window=2)
        # w[i, j] deberia ser 0 si |i - j| > 2
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 2:
                    self.assertEqual(w[i, j], 0.0)

    def test_shape(self):
        Q = np.random.default_rng(0).standard_normal((6, 4))
        K = np.random.default_rng(1).standard_normal((6, 4))
        V = np.random.default_rng(2).standard_normal((6, 4))
        out, _ = main.sliding_window_attention(Q, K, V, window=2)
        self.assertEqual(out.shape, (6, 4))


class TestGlobalLocal(unittest.TestCase):
    def test_global_tokens_ven_todo(self):
        n = 6
        Q = np.random.default_rng(0).standard_normal((n, 4))
        K = np.random.default_rng(1).standard_normal((n, 4))
        V = np.random.default_rng(2).standard_normal((n, 4))
        _, w = main.global_local_attention(Q, K, V, n_global=1, window=2)
        # w[0, :] (global token 0) deberia tener prob > 0 en todas las cols
        for j in range(n):
            self.assertGreater(w[0, j], 0)


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