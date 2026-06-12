"""Pruebas para 08-inference-metrics-goodput."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMetrics(unittest.TestCase):
    def test_record(self):
        m = main.InferenceMetrics()
        m.record_request(0.1, 0.05, 100)
        self.assertEqual(m.total_requests(), 1)

    def test_avg_ttft(self):
        m = main.InferenceMetrics()
        m.record_request(0.1, 0.05, 100)
        m.record_request(0.2, 0.05, 100)
        self.assertAlmostEqual(m.avg_ttft(), 0.15, places=4)

    def test_avg_tpot(self):
        m = main.InferenceMetrics()
        m.record_request(0.1, 0.05, 100)
        m.record_request(0.1, 0.06, 100)
        self.assertAlmostEqual(m.avg_tpot(), 0.055, places=4)

    def test_tokens_per_second(self):
        m = main.InferenceMetrics()
        m.record_request(0.1, 0.05, 100)
        m.duration_seconds = 1.0
        self.assertAlmostEqual(m.tokens_per_second(), 100.0, places=4)

    def test_rps(self):
        m = main.InferenceMetrics()
        m.record_request(0.1, 0.05, 100)
        m.record_request(0.1, 0.05, 100)
        m.duration_seconds = 1.0
        self.assertAlmostEqual(m.requests_per_second(), 2.0, places=4)

    def test_goodput_all(self):
        m = main.InferenceMetrics()
        m.record_request(0.1, 0.05, 100)
        m.record_request(0.2, 0.05, 100)
        self.assertEqual(m.goodput(0.5, 0.1), 1.0)

    def test_goodput_partial(self):
        m = main.InferenceMetrics()
        m.record_request(0.1, 0.05, 100)
        m.record_request(1.0, 0.5, 100)
        self.assertEqual(m.goodput(0.5, 0.1), 0.5)

    def test_e2e_p99(self):
        m = main.InferenceMetrics()
        for i in range(100):
            m.record_request(0.1, 0.01, 100)
        p99 = m.e2e_latency_p99()
        self.assertGreater(p99, 0)


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