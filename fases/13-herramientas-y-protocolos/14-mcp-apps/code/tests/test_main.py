"""Pruebas para 14-mcp-apps."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestManifest(unittest.TestCase):
    def test_basic(self):
        m = main.make_app_manifest("app", "1.0.0", "desc", ["interactive"], [])
        self.assertEqual(m["name"], "app")
        self.assertEqual(m["version"], "1.0.0")


class TestComponents(unittest.TestCase):
    def test_text(self):
        c = main.make_text_component("hello")
        self.assertEqual(c["type"], "text")
        self.assertEqual(c["content"], "hello")

    def test_image(self):
        c = main.make_image_component("https://x", "alt")
        self.assertEqual(c["type"], "image")
        self.assertEqual(c["alt"], "alt")

    def test_button(self):
        c = main.make_button_component("Submit", "submit", "primary")
        self.assertEqual(c["type"], "button")
        self.assertEqual(c["label"], "Submit")
        self.assertEqual(c["style"], "primary")

    def test_form(self):
        c = main.make_form_component([{"name": "x", "type": "text"}])
        self.assertEqual(c["type"], "form")
        self.assertEqual(len(c["fields"]), 1)

    def test_chart(self):
        c = main.make_chart_component("line", {"x": [1, 2]})
        self.assertEqual(c["type"], "chart")
        self.assertEqual(c["chartType"], "line")


class TestRender(unittest.TestCase):
    def test_text(self):
        m = main.make_app_manifest("a", "1.0", "d", [], [main.make_text_component("hi")])
        out = main.render_app(m)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["rendered"], "hi")

    def test_button_no_action(self):
        m = main.make_app_manifest("a", "1.0", "d", [],
            [main.make_button_component("OK", "ok")])
        out = main.render_app(m)
        self.assertEqual(out[0]["label"], "OK")
        self.assertNotIn("executed", out[0])

    def test_button_with_action(self):
        m = main.make_app_manifest("a", "1.0", "d", [],
            [main.make_button_component("OK", "ok_action")])
        out = main.render_app(m, actions={"ok_action": lambda: "result"})
        self.assertEqual(out[0]["executed"], "result")


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