"""Pruebas para 23-otel-genai-conventions."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestAgentSpan(unittest.TestCase):
    def test_basic(self):
        s = main.AgentSpan("test", "agent1")
        s.end()
        d = s.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertEqual(d["agent_name"], "agent1")
        self.assertIsNotNone(d["duration_ms"])

    def test_attributes(self):
        s = main.AgentSpan("test", "a")
        s.set_attribute("k", "v")
        s.end()
        self.assertEqual(s.attributes["k"], "v")

    def test_event(self):
        s = main.AgentSpan("test", "a")
        s.add_event("e1", {"x": 1})
        s.end()
        self.assertEqual(len(s.events), 1)


class TestRecordAgentRun(unittest.TestCase):
    def test_basic(self):
        s = main.AgentSpan("run", "agent")
        main.record_agent_run(s, "input", "output", model="gpt-4o", n_steps=3)
        self.assertEqual(s.attributes["gen_ai.request.model"], "gpt-4o")
        self.assertEqual(s.attributes["gen_ai.agent.steps"], 3)
        self.assertEqual(len(s.events), 2)


class TestRecordToolCall(unittest.TestCase):
    def test_basic(self):
        s = main.AgentSpan("run", "a")
        main.record_tool_call(s, "foo", {"x": 1}, "result", 100)
        self.assertEqual(len(s.events), 1)
        self.assertEqual(s.events[0]["name"], "tool_call")


class TestRecordHandoff(unittest.TestCase):
    def test_basic(self):
        s = main.AgentSpan("run", "a")
        main.record_handoff(s, "a1", "a2", "reason")
        self.assertEqual(len(s.events), 1)
        self.assertEqual(s.events[0]["name"], "handoff")


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