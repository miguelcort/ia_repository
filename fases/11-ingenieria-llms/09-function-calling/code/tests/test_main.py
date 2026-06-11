"""Pruebas para 09-function-calling."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSchema(unittest.TestCase):
    def test_basic(self):
        s = main.tool_to_schema("search", "Search web", {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"]
        })
        self.assertEqual(s["type"], "function")
        self.assertEqual(s["function"]["name"], "search")


class TestFormat(unittest.TestCase):
    def test_format(self):
        msg = main.format_tool_call("search", {"q": "AI"}, call_id="c1")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "search")


class TestParse(unittest.TestCase):
    def test_parse(self):
        text = "Voy a buscar <tool>search(query='AI')</tool> y <tool>calc(x=2)</tool>"
        calls = main.parse_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "search")
        self.assertEqual(calls[1]["args"]["x"], 2)


class TestParallel(unittest.TestCase):
    def test_basic(self):
        calls = [{"name": "a"}, {"name": "b"}]
        results = main.parallel_tool_calls(calls)
        self.assertEqual(len(results), 2)


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