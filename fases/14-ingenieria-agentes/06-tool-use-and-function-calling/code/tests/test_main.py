"""Pruebas para 06-tool-use-and-function-calling."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestBuildToolDef(unittest.TestCase):
    def test_basic(self):
        t = main.build_tool_def("foo", "do foo", {"type": "object"})
        self.assertEqual(t["type"], "function")
        self.assertEqual(t["function"]["name"], "foo")


class TestParseToolCall(unittest.TestCase):
    def test_basic(self):
        name, args = main.parse_tool_call("add(a=2, b=3)")
        self.assertEqual(name, "add")
        self.assertEqual(args, {"a": 2, "b": 3})

    def test_string_arg(self):
        name, args = main.parse_tool_call("echo(msg='hi')")
        self.assertEqual(args, {"msg": "hi"})

    def test_no_args(self):
        name, args = main.parse_tool_call("foo")
        self.assertEqual(name, "foo")
        self.assertEqual(args, {})


class TestExecuteToolCall(unittest.TestCase):
    def test_basic(self):
        tools = {"add": lambda a, b: a + b}
        r = main.execute_tool_call("add(a=2, b=3)", tools)
        self.assertEqual(r, 5)

    def test_unknown_tool(self):
        r = main.execute_tool_call("missing(x=1)", {})
        self.assertIn("unknown tool", r)

    def test_retry(self):
        call_count = [0]
        def fail_first(a, b):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("transient")
            return a + b
        tools = {"add": fail_first}
        r = main.execute_tool_call("add(a=1, b=2)", tools, max_retries=3)
        self.assertEqual(r, 3)
        self.assertEqual(call_count[0], 2)

    def test_all_retries_fail(self):
        def always_fail(a, b):
            raise RuntimeError("permanent")
        tools = {"add": always_fail}
        r = main.execute_tool_call("add(a=1, b=2)", tools, max_retries=2)
        self.assertIn("retries", r)


class TestFormatToolResult(unittest.TestCase):
    def test_basic(self):
        s = main.format_tool_result("add", 5)
        self.assertIn("5", s)

    def test_truncate(self):
        long_result = "x" * 3000
        s = main.format_tool_result("foo", long_result, max_length=100)
        self.assertIn("truncated", s)
        self.assertLess(len(s), 250)

    def test_dict(self):
        s = main.format_tool_result("foo", {"key": "value"})
        self.assertIn("key", s)


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