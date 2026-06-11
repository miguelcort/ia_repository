"""Pruebas para 05-tool-schema-design."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestScore(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(main.score_description(""), 0.0)

    def test_good(self):
        desc = ("Use this tool when the user asks about current weather. "
                "Example: get_weather({'city': 'NYC'}). "
                "Do not use for forecasts. ")
        score = main.score_description(desc)
        self.assertGreater(score, 0.7)

    def test_too_short(self):
        score = main.score_description("hi")
        self.assertLess(score, 0.5)


class TestName(unittest.TestCase):
    def test_valid(self):
        ok, _ = main.validate_name("get_weather")
        self.assertTrue(ok)

    def test_uppercase(self):
        ok, msg = main.validate_name("GetWeather")
        self.assertFalse(ok)
        self.assertIn("lowercase", msg)

    def test_special_char(self):
        ok, _ = main.validate_name("get-weather")
        self.assertFalse(ok)

    def test_underscore_start(self):
        ok, _ = main.validate_name("_foo")
        self.assertFalse(ok)


class TestConflict(unittest.TestCase):
    def test_no_conflict(self):
        ok, _ = main.check_name_conflict([{"name": "a"}], "b")
        self.assertTrue(ok)

    def test_conflict(self):
        ok, msg = main.check_name_conflict([{"name": "a"}], "a")
        self.assertFalse(ok)
        self.assertIn("conflict", msg)


class TestBuildDoc(unittest.TestCase):
    def test_basic(self):
        doc = main.build_tool_doc("foo", "desc", {"type": "object"})
        self.assertEqual(doc["name"], "foo")

    def test_full(self):
        doc = main.build_tool_doc(
            "foo", "desc", {"type": "object"},
            when_to_use="x", when_not_to_use="y", examples="z",
        )
        self.assertIn("When to use", doc["description"])
        self.assertIn("When NOT", doc["description"])
        self.assertIn("Examples", doc["description"])


class TestLint(unittest.TestCase):
    def test_clean(self):
        tools = [
            {"name": "foo", "description": "Good description with examples. Use this when foo is needed."},
            {"name": "bar", "description": "Another good tool with examples. Use this when bar is needed."},
        ]
        issues = main.lint_tools(tools)
        self.assertEqual(issues, [])

    def test_bad_name(self):
        tools = [{"name": "BadName", "description": "desc"}]
        issues = main.lint_tools(tools)
        self.assertGreater(len(issues), 0)

    def test_duplicate(self):
        tools = [
            {"name": "foo", "description": "Good description with examples."},
            {"name": "foo", "description": "Another good description with examples."},
        ]
        issues = main.lint_tools(tools)
        self.assertTrue(any("duplicate" in i for i in issues))


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