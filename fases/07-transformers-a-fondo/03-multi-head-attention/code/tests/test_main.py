"""Pruebas para 03-multi-head-attention."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSplitMerge(unittest.TestCase):
    def test_split_preserva_datos(self):
        # split -> merge deberia preservar datos
        rng = np.random.default_rng(0)
        x = rng.standard_normal((4, 8))
        splitted = main.split_heads(x, 2, 4)
        merged = main.merge_heads(splitted)
        np.testing.assert_allclose(merged, x, atol=1e-9)

    def test_shapes(self):
        x = np.zeros((6, 12))
        s = main.split_heads(x, 3, 4)
        self.assertEqual(s.shape, (3, 6, 4))
        m = main.merge_heads(s)
        self.assertEqual(m.shape, (6, 12))


class TestMultiHead(unittest.TestCase):
    def test_output_shape(self):
        seq, d, n = 4, 8, 2
        X = np.random.default_rng(1).standard_normal((seq, d))
        W_Q, W_K, W_V, W_O = main.init_params(d, seed=1)
        out = main.multi_head_attention(X, W_Q, W_K, W_V, W_O, n)
        self.assertEqual(out.shape, (seq, d))

    def test_cada_head_independiente(self):
        # Cambiar W_O no deberia cambiar suma de outputs (antes de O)
        seq, d, n = 4, 8, 2
        X = np.random.default_rng(2).standard_normal((seq, d))
        W_Q, W_K, W_V, W_O = main.init_params(d, seed=2)
        out1 = main.multi_head_attention(X, W_Q, W_K, W_V, W_O, n)
        out2 = main.multi_head_attention(X, W_Q, W_K, W_V, W_O * 2, n)
        # No es la misma, pero misma shape y normas proporcionales
        self.assertEqual(out1.shape, out2.shape)
        # 2 * W_O -> output * 2
        np.testing.assert_allclose(out2, out1 * 2, atol=1e-9)

    def test_d_model_debe_dividir(self):
        # d_model=8, n_heads=3 -> 8%3 != 0 -> assert fail
        X = np.zeros((2, 8))
        W_Q = W_K = W_V = W_O = np.eye(8)
        with self.assertRaises(AssertionError):
            main.multi_head_attention(X, W_Q, W_K, W_V, W_O, n_heads=3)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("output shape", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()