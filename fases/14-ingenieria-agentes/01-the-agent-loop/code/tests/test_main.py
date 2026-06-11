"""Pruebas para 01-the-agent-loop."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestAgentLoop(unittest.TestCase):
    def test_step_with_tool(self):
        def llm(obs, tools):
            return "thinking", ("foo", {})
        tools = {"foo": lambda: "result"}
        agent = main.AgentLoop(llm, tools, max_iterations=3)
        thought, result = agent.step("obs")
        self.assertEqual(thought, "thinking")
        self.assertEqual(result, "result")

    def test_step_finish(self):
        def llm(obs, tools):
            return "done", None
        tools = {}
        agent = main.AgentLoop(llm, tools, max_iterations=3)
        thought, result = agent.step("obs")
        self.assertIsNone(result)
        self.assertEqual(agent.history[-1]["type"], "finish")

    def test_step_unknown_tool(self):
        def llm(obs, tools):
            return "thinking", ("missing", {})
        tools = {}
        agent = main.AgentLoop(llm, tools, max_iterations=3)
        thought, result = agent.step("obs")
        self.assertIsNone(result)

    def test_run_max_iterations(self):
        call_count = [0]
        def llm(obs, tools):
            call_count[0] += 1
            return "thinking", ("foo", {})
        tools = {"foo": lambda: "ok"}
        agent = main.AgentLoop(llm, tools, max_iterations=3)
        result = agent.run("test")
        self.assertIsNone(result)
        self.assertEqual(call_count[0], 3)

    def test_run_to_finish(self):
        def llm(obs, tools):
            if call_count[0] == 0:
                return "first", ("foo", {})
            return "done", None
        call_count = [0]
        def llm2(obs, tools):
            if call_count[0] == 0:
                call_count[0] += 1
                return "first", ("foo", {})
            return "done", None
        tools = {"foo": lambda: "ok"}
        agent = main.AgentLoop(llm2, tools, max_iterations=5)
        result = agent.run("test")
        self.assertEqual(result, "done")


class TestReactPrompt(unittest.TestCase):
    def test_basic(self):
        prompt = main.react_prompt("obs", {"foo": lambda: 1})
        self.assertIn("obs", prompt)
        self.assertIn("foo", prompt)


class TestParseReact(unittest.TestCase):
    def test_basic(self):
        text = "Thought: I should do X\nAction: foo(bar=1)"
        thought, action = main.parse_react_output(text)
        self.assertEqual(thought, "I should do X")
        self.assertEqual(action, ("foo", {"bar": "1"}))

    def test_thought_only(self):
        text = "Thought: done"
        thought, action = main.parse_react_output(text)
        self.assertEqual(thought, "done")
        self.assertIsNone(action)

    def test_no_action(self):
        text = ""
        thought, action = main.parse_react_output(text)
        self.assertIsNone(thought)


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