"""Pruebas para 12-optimizacion-de-inferencia."""
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
    def test_basic(self):
        mem = main.kv_cache_memory(2, 2, 4, 8, 1, num_bytes=2)
        # 2 * 2 * 2 * 8 * 4 * 2 * 1 = 512
        self.assertEqual(mem, 512)

    def test_int8(self):
        mem = main.kv_cache_memory(2, 2, 4, 8, 1, num_bytes=1)
        # Half
        self.assertEqual(mem, 256)


class TestPaged(unittest.TestCase):
    def test_blocks(self):
        n = main.paged_attention_blocks(total_tokens=100, block_size=16)
        # ceil(100/16) = 7
        self.assertEqual(n, 7)

    def test_perfect(self):
        n = main.paged_attention_blocks(64, block_size=16)
        self.assertEqual(n, 4)


class TestContinuousBatching(unittest.TestCase):
    def test_saturated(self):
        t = main.continuous_batching_throughput(10, 8, 100)
        # max_batch_size = 8
        self.assertEqual(t, 800)


class TestPrefixCache(unittest.TestCase):
    def test_shared(self):
        # 4 requests con mismo prefix
        rate = main.prefix_cache_hit_rate([100, 100, 100, 100], 4)
        # 3 sequences de 100 compartidas con la primera
        self.assertEqual(rate, 0.75)


class TestChunked(unittest.TestCase):
    def test_basic(self):
        # 2 sequences de 1000, chunk 512: 2 + 2 = 4
        n = main.chunked_prefill_schedule([1000, 1000], chunk_size=512)
        self.assertEqual(n, 4)


class TestOptimizations(unittest.TestCase):
    def test_ocho(self):
        o = main.inference_optimizations()
        self.assertEqual(len(o), 8)


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