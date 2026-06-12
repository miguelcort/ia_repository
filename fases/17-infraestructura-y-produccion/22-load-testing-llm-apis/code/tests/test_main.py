"""Pruebas para 22-load-testing-llm-apis."""
from __future__ import annotations
import random
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestLoad(unittest.TestCase):
    def test_simulate(self):
        random.seed(42)
        lt = main.LoadTest()
        for _ in range(10):
            lt.simulate_request(base_latency=0.5, error_rate=0.0)
        self.assertEqual(lt.requests, 10)
        self.assertEqual(lt.errors, 0)
        self.assertEqual(len(lt.latencies), 10)

    def test_with_errors(self):
        random.seed(42)
        lt = main.LoadTest()
        for _ in range(100):
            lt.simulate_request(base_latency=0.5, error_rate=0.1)
        self.assertGreater(lt.errors, 0)

    def test_percentile(self):
        random.seed(42)
        lt = main.LoadTest()
        for _ in range(100):
            lt.simulate_request(base_latency=0.5, error_rate=0.0)
        p50 = lt.percentile(50)
        self.assertGreater(p50, 0)
        p99 = lt.percentile(99)
        self.assertGreaterEqual(p99, p50)

    def test_error_rate(self):
        lt = main.LoadTest()
        self.assertEqual(lt.error_rate(), 0.0)

    def test_throughput(self):
        lt = main.LoadTest()
        lt.requests = 100
        self.assertEqual(lt.throughput_rps(2), 50.0)


class TestRamp(unittest.TestCase):
    def test_ramp(self):
        random.seed(42)
        lt = main.LoadTest()
        duration = main.ramp_load(lt, max_concurrent=3, duration=1, base_latency=0.1)
        self.assertGreater(lt.requests, 0)
        self.assertGreater(duration, 0)


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