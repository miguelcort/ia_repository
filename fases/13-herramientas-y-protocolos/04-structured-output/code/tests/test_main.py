"""Pruebas para 04-structured-output."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestJsonSchemaForResponse(unittest.TestCase):
    def test_basic(self):
        schema = {"name": "x", "schema": {"type": "object"}}
        fmt = main.json_schema_for_response(schema)
        self.assertEqual(fmt["type"], "json_schema")
        self.assertEqual(fmt["json_schema"]["name"], "x")
        self.assertTrue(fmt["json_schema"]["strict"])


class TestJsonMode(unittest.TestCase):
    def test_format(self):
        fmt = main.json_mode()
        self.assertEqual(fmt["type"], "json_object")


class TestParseStrictJson(unittest.TestCase):
    def test_valid(self):
        data, msg = main.parse_strict_json('{"a": 1, "b": 2}', {"required": ["a"]})
        self.assertIsNotNone(data)
        self.assertEqual(data["a"], 1)

    def test_invalid_json(self):
        data, msg = main.parse_strict_json("not json", {"required": ["a"]})
        self.assertIsNone(data)

    def test_missing_required(self):
        data, msg = main.parse_strict_json('{"b": 2}', {"required": ["a"]})
        self.assertIsNone(data)
        self.assertIn("missing", msg)


class TestPydanticToSchema(unittest.TestCase):
    def test_basic(self):
        model = {
            "fields": {
                "name": {"type": "string", "required": True, "description": "name"},
                "age": {"type": "integer", "required": True},
            }
        }
        schema = main.pydantic_to_schema(model)
        self.assertIn("name", schema["properties"])
        self.assertIn("name", schema["required"])
        self.assertIn("age", schema["required"])
        self.assertFalse(schema["additionalProperties"])

    def test_optional(self):
        model = {
            "fields": {
                "a": {"type": "string", "required": True},
                "b": {"type": "string", "required": False},
            }
        }
        schema = main.pydantic_to_schema(model)
        self.assertIn("a", schema["required"])
        self.assertNotIn("b", schema["required"])


class TestInstructor(unittest.TestCase):
    def test_wrap(self):
        r = main.instructor_wrap(None, "gpt-4o", {"type": "object"}, [])
        self.assertIn("model", r)


class TestOutlines(unittest.TestCase):
    def test_constrained(self):
        r = main.outlines_constrained_decode("test", {"type": "object"}, "model")
        self.assertTrue(r["constrained"])


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