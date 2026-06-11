"""Pruebas para 10-mcp-resources-and-prompts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestResourceURI(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(main.validate_resource_uri("file://docs/readme.md"))
        self.assertTrue(main.validate_resource_uri("https://example.com/data"))

    def test_invalid(self):
        self.assertFalse(main.validate_resource_uri("no-scheme"))
        self.assertFalse(main.validate_resource_uri("file:/no-slash"))


class TestMakeResource(unittest.TestCase):
    def test_basic(self):
        r = main.make_resource("file://x.md", "x", "desc", "text/markdown")
        self.assertEqual(r["uri"], "file://x.md")
        self.assertEqual(r["mimeType"], "text/markdown")

    def test_invalid_uri(self):
        with self.assertRaises(ValueError):
            main.make_resource("invalid", "x", "desc")


class TestResourceRead(unittest.TestCase):
    def test_read(self):
        server = {"resources": {"file://x.md": {"content": "hello", "mimeType": "text/plain"}}}
        r = main.resource_read(server, "file://x.md")
        self.assertEqual(r["contents"][0]["text"], "hello")

    def test_not_found(self):
        r = main.resource_read({}, "file://x.md")
        self.assertIn("error", r)


class TestMakePrompt(unittest.TestCase):
    def test_basic(self):
        p = main.make_prompt("foo", "do {x}", [{"name": "x", "required": True}])
        self.assertEqual(p["name"], "foo")
        self.assertEqual(len(p["arguments"]), 1)

    def test_no_args(self):
        p = main.make_prompt("foo", "desc")
        self.assertEqual(p["arguments"], [])


class TestRenderPrompt(unittest.TestCase):
    def test_render(self):
        p = main.make_prompt("foo", "Hello {name}", [{"name": "name", "required": True}])
        r = main.render_prompt(p, {"name": "World"})
        self.assertEqual(r["messages"][0]["content"], "Hello World")

    def test_missing_arg(self):
        p = main.make_prompt("foo", "Hello {name}", [{"name": "name", "required": True}])
        r = main.render_prompt(p, {})
        # missing arg -> empty string
        self.assertIn("Hello", r["messages"][0]["content"])


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