"""Pruebas para 07-building-an-mcp-server."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = main.ToolRegistry()

    def test_register(self):
        @self.reg.tool("foo", "do foo", {"type": "object"})
        def foo():
            return 1
        self.assertIn("foo", self.reg.tools)

    def test_list(self):
        @self.reg.tool("foo", "do foo", {"type": "object"})
        def foo():
            return 1
        tools = self.reg.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "foo")

    def test_call(self):
        @self.reg.tool("add", "add", {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        })
        def add(a, b):
            return a + b
        result = self.reg.call("add", {"a": 2, "b": 3})
        self.assertEqual(result["result"], 5)

    def test_call_not_found(self):
        result = self.reg.call("missing", {})
        self.assertIn("error", result)


class TestResourceRegistry(unittest.TestCase):
    def test_register_and_list(self):
        reg = main.ResourceRegistry()
        @reg.resource("file://test", "test file")
        def test():
            return "content"
        self.assertIn("file://test", reg.resources)
        resources = reg.list_resources()
        self.assertEqual(len(resources), 1)

    def test_read(self):
        reg = main.ResourceRegistry()
        @reg.resource("file://test", "test file")
        def test():
            return "content"
        result = reg.read("file://test")
        self.assertEqual(result["contents"], "content")


class TestPromptRegistry(unittest.TestCase):
    def test_get(self):
        reg = main.PromptRegistry()
        @reg.prompt("greet", "greeting", [{"name": "who", "required": True}])
        def greet(who):
            return [{"role": "user", "content": f"Hello {who}"}]
        result = reg.get("greet", {"who": "World"})
        self.assertEqual(result["messages"][0]["content"], "Hello World")


class TestBuildServer(unittest.TestCase):
    def test_build(self):
        server = main.build_mcp_server("test", "1.0.0")
        self.assertEqual(server["name"], "test")
        self.assertEqual(server["protocolVersion"], "2024-11-05")


class TestHandleRequest(unittest.TestCase):
    def test_tools_list(self):
        server = main.build_mcp_server("test")
        @server["tools"].tool("foo", "do foo", {"type": "object"})
        def foo():
            return 1
        req = main.handle_request(server, {"method": "tools/list"})
        self.assertIn("tools", req["result"])

    def test_unknown_method(self):
        server = main.build_mcp_server("test")
        req = main.handle_request(server, {"method": "unknown"})
        self.assertIn("error", req)


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