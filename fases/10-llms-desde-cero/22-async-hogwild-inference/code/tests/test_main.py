"""Pruebas para 22-async-hogwild-inference."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestAsyncBatch(unittest.TestCase):
    def test_all_processed(self):
        requests = [(i, f"p{i}") for i in range(10)]
        results = main.async_batch_simulation(requests, batch_size=4)
        self.assertEqual(len(results), 10)


class TestThroughput(unittest.TestCase):
    def test_sync(self):
        t = main.throughput_synchronous(100, latency_per_request=100)
        # 100 / (100 * 0.1) = 10 req/sec
        self.assertAlmostEqual(t, 10.0, places=4)

    def test_async(self):
        # Async deberia ser mayor
        t_sync = main.throughput_synchronous(100, latency_per_request=100)
        t_async = main.throughput_async(100, batch_size=4, latency_per_batch=120)
        self.assertGreater(t_async, t_sync)


class TestHogwild(unittest.TestCase):
    def test_basic(self):
        t = main.hogwild_throughput(n_workers=4, n_requests=100, latency=100)
        # 4 / 0.1 * 100 / 4 = 1000
        self.assertGreater(t, 0)


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