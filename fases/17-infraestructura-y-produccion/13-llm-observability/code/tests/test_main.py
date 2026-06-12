"""Pruebas para 13-llm-observability."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSpan(unittest.TestCase):
    def test_create(self):
        s = main.Span("test")
        self.assertEqual(s.name, "test")
        self.assertIsNotNone(s.id)
        self.assertIsNone(s.parent_id)
        self.assertIsNone(s.end)

    def test_attribute(self):
        s = main.Span("test")
        s.set_attribute("model", "gpt-4o")
        self.assertEqual(s.attributes["model"], "gpt-4o")

    def test_event(self):
        s = main.Span("test")
        s.add_event("retry")
        self.assertEqual(len(s.events), 1)

    def test_duration(self):
        s = main.Span("test")
        s.finish()
        d = s.duration()
        self.assertGreaterEqual(d, 0)


class TestTracer(unittest.TestCase):
    def test_start_span(self):
        t = main.Tracer()
        s = t.start_span("a")
        self.assertIn(s.id, t.spans)

    def test_get_trace(self):
        t = main.Tracer()
        s = t.start_span("a")
        child = t.start_span("b", parent_id=s.id)
        trace = t.get_trace(s.id)
        self.assertEqual(len(trace), 2)


class TestMetrics(unittest.TestCase):
    def test_increment(self):
        m = main.Metrics()
        m.increment("requests")
        m.increment("requests", 5)
        self.assertEqual(m.get_counter("requests"), 6)

    def test_observe(self):
        m = main.Metrics()
        m.observe("latency", 0.1)
        m.observe("latency", 0.2)
        stats = m.get_histogram_stats("latency")
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["avg"], 0.15, places=2)

    def test_empty(self):
        m = main.Metrics()
        self.assertEqual(m.get_counter("missing"), 0)
        self.assertIsNone(m.get_histogram_stats("missing"))


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