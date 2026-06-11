"""Pruebas para 03-outputs-estructurados."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestExtractJSON(unittest.TestCase):
    def test_basic(self):
        text = 'Aqui esta: {"name": "test", "value": 42}'
        obj = main.extract_json(text)
        self.assertEqual(obj["name"], "test")
        self.assertEqual(obj["value"], 42)

    def test_no_json(self):
        text = "no json aqui"
        self.assertIsNone(main.extract_json(text))


class TestSchemaValidate(unittest.TestCase):
    def test_valid(self):
        obj = {"name": "test", "value": 42}
        valid = main.json_schema_validate(obj, {"name": str, "value": int})
        self.assertTrue(valid)

    def test_missing_field(self):
        obj = {"name": "test"}
        valid = main.json_schema_validate(obj, {"name": str, "value": int})
        self.assertFalse(valid)

    def test_wrong_type(self):
        obj = {"name": "test", "value": "string"}
        valid = main.json_schema_validate(obj, {"name": str, "value": int})
        self.assertFalse(valid)


class TestFunctionCall(unittest.TestCase):
    def test_format(self):
        s = main.function_call_format("search", {"q": "AI"})
        d = eval(s) if not __import__("json").loads else __import__("json").loads(s)
        self.assertEqual(d["name"], "search")
        self.assertEqual(d["arguments"]["q"], "AI")


class TestParseToolCalls(unittest.TestCase):
    def test_parse(self):
        text = "Aqui esta el codigo: ```python\nprint('hi')\n```"
        calls = main.parse_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "python")


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