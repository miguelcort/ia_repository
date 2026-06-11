"""Pruebas para 14-model-context-protocol."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestRequest(unittest.TestCase):
    def test_format(self):
        r = main.mcp_request("tools/list", {"foo": "bar"}, id=1)
        self.assertEqual(r["jsonrpc"], "2.0")
        self.assertEqual(r["method"], "tools/list")
        self.assertEqual(r["params"]["foo"], "bar")


class TestResponse(unittest.TestCase):
    def test_format(self):
        r = main.mcp_response({"status": "ok"})
        self.assertEqual(r["jsonrpc"], "2.0")
        self.assertIn("result", r)


class TestError(unittest.TestCase):
    def test_format(self):
        r = main.mcp_error(-32601, "Method not found")
        self.assertEqual(r["error"]["code"], -32601)


class TestToolDef(unittest.TestCase):
    def test_basic(self):
        t = main.mcp_tool_def("search", "Search", {"type": "object"})
        self.assertEqual(t["name"], "search")
        self.assertIn("inputSchema", t)


class TestComponents(unittest.TestCase):
    def test_four(self):
        c = main.mcp_components()
        self.assertEqual(len(c), 4)
        self.assertIn("Resources", c)
        self.assertIn("Tools", c)


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