"""Pruebas para 02-function-calling-deep-dive."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import json


class TestToolCallPayload(unittest.TestCase):
    def test_basic(self):
        tc = main.tool_call_payload("foo", {"x": 1})
        self.assertEqual(tc["function"]["name"], "foo")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args, {"x": 1})

    def test_id(self):
        tc = main.tool_call_payload("foo", {}, tool_call_id="abc")
        self.assertEqual(tc["id"], "abc")


class TestAssistantMessage(unittest.TestCase):
    def test_with_tool_calls(self):
        msg = main.assistant_message_with_tool_calls(
            [main.tool_call_payload("foo", {})]
        )
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(len(msg["tool_calls"]), 1)


class TestToolResult(unittest.TestCase):
    def test_string_content(self):
        msg = main.tool_result_message("call_1", "result")
        self.assertEqual(msg["role"], "tool")
        self.assertEqual(msg["content"], "result")

    def test_dict_content(self):
        msg = main.tool_result_message("call_1", {"key": "value"})
        self.assertEqual(json.loads(msg["content"]), {"key": "value"})


class TestConversation(unittest.TestCase):
    def test_full(self):
        msgs = main.conversation_with_tools(
            [{"role": "user", "content": "test"}],
            tools=[],
            tool_calls=[main.tool_call_payload("foo", {}, "call_1")],
            results=["result"],
        )
        # user + assistant + tool = 3
        self.assertEqual(len(msgs), 3)


class TestParseArgs(unittest.TestCase):
    def test_parse(self):
        tc = main.tool_call_payload("foo", {"a": 1, "b": [2, 3]})
        args = main.parse_tool_arguments(tc)
        self.assertEqual(args["a"], 1)
        self.assertEqual(args["b"], [2, 3])


class TestToolChoice(unittest.TestCase):
    def test_auto(self):
        self.assertEqual(main.tool_choice_auto(), "auto")

    def test_required(self):
        self.assertEqual(main.tool_choice_required(), "required")

    def test_none(self):
        self.assertEqual(main.tool_choice_none(), "none")

    def test_specific(self):
        c = main.tool_choice_specific("my_tool")
        self.assertEqual(c["function"]["name"], "my_tool")


class TestStrictMode(unittest.TestCase):
    def test_all_required(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
            },
        }
        strict = main.strict_mode_schema(schema)
        self.assertIn("a", strict["required"])
        self.assertIn("b", strict["required"])
        self.assertFalse(strict["additionalProperties"])


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