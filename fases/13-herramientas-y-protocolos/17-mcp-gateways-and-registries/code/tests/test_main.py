"""Pruebas para 17-mcp-gateways-and-registries."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = main.MCPRegistry()

    def test_register(self):
        self.reg.register("a", "url", "1.0", ["tools"])
        self.assertIn("a", self.reg.servers)

    def test_lookup(self):
        self.reg.register("a", "url", "1.0", ["tools"])
        self.reg.register("b", "url", "1.0", ["resources"])
        tools = self.reg.lookup("tools")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "a")

    def test_list(self):
        self.reg.register("a", "url", "1.0", ["tools"])
        self.reg.register("b", "url", "1.0", ["tools"])
        self.assertEqual(len(self.reg.list_servers()), 2)

    def test_unregister(self):
        self.reg.register("a", "url", "1.0", ["tools"])
        self.reg.unregister("a")
        self.assertNotIn("a", self.reg.servers)


class TestGateway(unittest.TestCase):
    def setUp(self):
        self.reg = main.MCPRegistry()
        self.reg.register("a", "url", "1.0", ["tools"])
        self.reg.register("b", "url", "1.0", ["tools"])
        self.gw = main.MCPGateway(self.reg)

    def test_route(self):
        res = self.gw.route("tools", lambda s: "ok")
        self.assertIsNotNone(res)
        self.assertIn("server", res)

    def test_route_no_match(self):
        res = self.gw.route("nonexistent", lambda s: "ok")
        self.assertIsNone(res)
        self.assertEqual(self.gw.metrics["errors"], 1)

    def test_route_round_robin(self):
        for _ in range(3):
            self.gw.route("tools", lambda s: s["name"])
        self.assertEqual(self.gw.metrics["requests"], 3)

    def test_forward(self):
        res = self.gw.forward("a", "test")
        self.assertEqual(res["server"], "a")
        self.assertIn("test", res["result"])

    def test_forward_unknown(self):
        res = self.gw.forward("missing", "test")
        self.assertIsNone(res)


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