"""Pruebas para 01-the-tool-interface."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMakeTool(unittest.TestCase):
    def test_basic(self):
        t = main.make_tool("foo", "do foo", {"type": "object"})
        self.assertEqual(t["name"], "foo")
        self.assertEqual(t["description"], "do foo")

    def test_default_returns(self):
        t = main.make_tool("foo", "do foo", {"type": "object"})
        self.assertEqual(t["returns"], {"type": "string"})


class TestValidateParameters(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "n": {"type": "integer"},
                "ratio": {"type": "number"},
                "active": {"type": "boolean"},
            },
            "required": ["city"],
        }

    def test_valid(self):
        ok, msg = main.validate_parameters({"city": "NYC", "n": 5}, self.schema)
        self.assertTrue(ok)

    def test_missing_required(self):
        ok, msg = main.validate_parameters({}, self.schema)
        self.assertFalse(ok)
        self.assertIn("missing", msg)

    def test_wrong_type(self):
        ok, msg = main.validate_parameters({"city": 123}, self.schema)
        self.assertFalse(ok)
        self.assertIn("must be string", msg)

    def test_unknown_param(self):
        ok, msg = main.validate_parameters({"city": "NYC", "foo": 1}, self.schema)
        self.assertFalse(ok)


class TestSerialize(unittest.TestCase):
    def test_format(self):
        t = main.make_tool("get_weather", "Get weather", {"type": "object"})
        s = main.serialize_tool(t)
        self.assertEqual(s["type"], "function")
        self.assertEqual(s["function"]["name"], "get_weather")


class TestRouter(unittest.TestCase):
    def test_found(self):
        t = main.make_tool("foo", "do foo", {"type": "object", "properties": {}})
        r = main.tool_router([t], {"name": "foo", "arguments": {}})
        self.assertEqual(r["status"], "ok")

    def test_not_found(self):
        r = main.tool_router([], {"name": "missing", "arguments": {}})
        self.assertEqual(r["status"], "error")


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