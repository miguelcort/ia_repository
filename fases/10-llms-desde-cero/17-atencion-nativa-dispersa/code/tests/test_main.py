"""Pruebas para 17-atencion-nativa-dispersa."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDense(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        Q, K, V = [rng.standard_normal((8, 16)) for _ in range(3)]
        out = main.dense_attention(Q, K, V)
        self.assertEqual(out.shape, (8, 16))


class TestSlidingWindow(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        Q, K, V = [rng.standard_normal((8, 16)) for _ in range(3)]
        out = main.sliding_window_attention(Q, K, V, window=2)
        self.assertEqual(out.shape, (8, 16))


class TestCompressed(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        Q, K, V = [rng.standard_normal((8, 16)) for _ in range(3)]
        out = main.compressed_attention(Q, K, V, block_size=2)
        self.assertEqual(out.shape, (8, 16))


class TestMemoryComplexity(unittest.TestCase):
    def test_dense(self):
        self.assertEqual(main.memory_complexity(1024, "dense"), 1024 * 1024)

    def test_sliding(self):
        # n * window
        self.assertEqual(main.memory_complexity(1024, "sliding", window=64), 1024 * 64)

    def test_compressed(self):
        # n * (n / block_size)
        self.assertEqual(main.memory_complexity(1024, "compressed", block_size=4), 1024 * 256)


class TestComponents(unittest.TestCase):
    def test_seis(self):
        c = main.nsa_components()
        self.assertEqual(len(c), 6)


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