"""Pruebas para 12-kv-cache-y-flash-attention."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestKVCache(unittest.TestCase):
    def test_init(self):
        c = main.KVCache(n_layers=2, n_heads=4, max_seq=128, d_k=16)
        self.assertEqual(c.n_layers, 2)
        self.assertEqual(c.cur_len, 0)

    def test_update_y_get(self):
        c = main.KVCache(n_layers=1, n_heads=2, max_seq=10, d_k=4)
        k = np.random.default_rng(0).standard_normal((1, 2, 4))
        v = np.random.default_rng(1).standard_normal((1, 2, 4))
        c.update(layer=0, k_new=k, v_new=v)
        c.advance(1)
        k_out, v_out = c.get(layer=0)
        self.assertEqual(k_out.shape, (1, 2, 4))
        np.testing.assert_array_equal(k_out[0], k[0])

    def test_multi_tokens(self):
        c = main.KVCache(n_layers=1, n_heads=2, max_seq=10, d_k=4)
        for t in range(3):
            k = np.random.default_rng(t).standard_normal((1, 2, 4))
            v = np.random.default_rng(100 + t).standard_normal((1, 2, 4))
            c.update(layer=0, k_new=k, v_new=v)
            c.advance(1)
        k_out, v_out = c.get(layer=0)
        self.assertEqual(k_out.shape, (3, 2, 4))


class TestFlashAttention(unittest.TestCase):
    def test_vs_naive(self):
        # Mock deberia dar resultado similar a naive
        n = 32
        rng = np.random.default_rng(0)
        Q, K, V = [rng.standard_normal((n, 8)) for _ in range(3)]
        out_naive = main.naive_attention(Q, K, V)
        out_flash = main.flash_attention_demo(Q, K, V)
        # Tolerancia: bloques aproximan, no son exactos
        # El mock online softmax deberia ser exacto para bloques completos
        np.testing.assert_allclose(out_flash, out_naive, atol=1e-4)

    def test_shape(self):
        n = 16
        rng = np.random.default_rng(0)
        Q, K, V = [rng.standard_normal((n, 8)) for _ in range(3)]
        out = main.flash_attention_demo(Q, K, V)
        self.assertEqual(out.shape, (n, 8))


class TestMemoryComparison(unittest.TestCase):
    def test_naive_mas_memoria(self):
        naive, flash = main.memory_comparison(1024)
        # Naive: n^2, Flash: block^2
        self.assertGreater(naive, flash)


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