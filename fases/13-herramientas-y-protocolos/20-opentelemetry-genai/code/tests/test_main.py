"""Pruebas para 20-opentelemetry-genai."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSpan(unittest.TestCase):
    def test_basic(self):
        s = main.Span("test", "trace1")
        s.end()
        d = s.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertIsNotNone(d["duration_ms"])

    def test_attributes(self):
        s = main.Span("test", "trace1")
        s.set_attribute("k", "v")
        s.end()
        self.assertEqual(s.attributes["k"], "v")

    def test_events(self):
        s = main.Span("test", "trace1")
        s.add_event("e1", {"a": 1})
        s.end()
        self.assertEqual(len(s.events), 1)


class TestTracer(unittest.TestCase):
    def test_start_span(self):
        t = main.Tracer()
        s = t.start_span("foo")
        self.assertEqual(s.name, "foo")

    def test_trace_id_consistent(self):
        t = main.Tracer()
        s1 = t.start_span("a")
        s2 = t.start_span("b")
        # second span should reuse trace_id from first
        self.assertEqual(s1.trace_id, s2.trace_id)


class TestGenAIAttributes(unittest.TestCase):
    def test_basic(self):
        attrs = main.gen_ai_attributes("gpt-4o", "openai", 100, 50)
        self.assertEqual(attrs["gen_ai.system"], "openai")
        self.assertEqual(attrs["gen_ai.usage.input_tokens"], 100)
        self.assertEqual(attrs["gen_ai.usage.output_tokens"], 50)


class TestRecordTool(unittest.TestCase):
    def test_basic(self):
        s = main.Span("test", "trace1")
        main.record_tool_span(s, "foo", {"x": 1}, {"y": 2})
        self.assertEqual(len(s.events), 2)
        self.assertEqual(s.events[0]["name"], "tool_call")
        self.assertEqual(s.events[1]["name"], "tool_result")


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