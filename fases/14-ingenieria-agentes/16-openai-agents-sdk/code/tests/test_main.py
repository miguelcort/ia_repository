"""Pruebas para 16-openai-agents-sdk."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestAgent(unittest.TestCase):
    def test_basic(self):
        a = main.Agent("test", "instructions")
        self.assertEqual(a.name, "test")
        self.assertEqual(a.model, "gpt-4o")

    def test_run(self):
        a = main.Agent("test", "instructions")
        r = a.run("hello")
        self.assertIn("response", r)
        self.assertEqual(r["agent"], "test")

    def test_handoff(self):
        a1 = main.Agent("a1", "sys1")
        a2 = main.Agent("a2", "sys2")
        a1.add_handoff(a2)
        r = a1.run("transfer me to a2")
        self.assertEqual(r["handoff_to"], "a2")
        self.assertEqual(a1.handoff_count, 1)


class TestRunner(unittest.TestCase):
    def test_basic(self):
        a = main.Agent("test", "instructions")
        runner = main.Runner(a, tracing=False)
        results = runner.run("hello")
        self.assertEqual(len(results), 1)

    def test_handoff_chain(self):
        a1 = main.Agent("a1", "sys1")
        a2 = main.Agent("a2", "sys2")
        a1.add_handoff(a2)
        runner = main.Runner(a1, tracing=True)
        results = runner.run("transfer to a2")
        # 2 turns: a1 handoffs, a2 responds
        self.assertEqual(len(results), 2)
        # a1 then a2
        self.assertEqual(runner.traces[0]["agent"], "a1")
        self.assertEqual(runner.traces[1]["agent"], "a2")
        self.assertGreaterEqual(len(runner.traces), 2)


class TestGuardrail(unittest.TestCase):
    def test_basic(self):
        g = main.Guardrail("test", lambda x: "ok" in x)
        self.assertTrue(g.validate("this is ok"))
        self.assertFalse(g.validate("this is bad"))


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