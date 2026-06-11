"""Pruebas para 23-capstone-tool-ecosystem."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestEcosystem(unittest.TestCase):
    def setUp(self):
        self.eco = main.ToolEcosystem()

    def test_register(self):
        self.eco.register_tool("foo", lambda: 1)
        self.assertIn("foo", self.eco.tools)

    def test_route_capability(self):
        self.eco.register_tool("a", lambda: 1, cost=5.0, capability="search")
        self.eco.register_tool("b", lambda: 1, cost=1.0, capability="search")
        t = self.eco.route("search")
        self.assertEqual(t["name"], "b")

    def test_route_no_match(self):
        self.assertIsNone(self.eco.route("nonexistent"))

    def test_call_basic(self):
        self.eco.register_tool("foo", lambda x: x * 2)
        result = self.eco.call("foo", {"x": 5})
        self.assertEqual(result, 10)

    def test_call_with_fallback(self):
        self.eco.register_tool("a", lambda x: 1 / 0)
        self.eco.register_tool("b", lambda x: "ok")
        result = self.eco.call("a", {"x": 1}, fallback_chain=["b"])
        self.assertEqual(result, "ok")
        self.assertEqual(self.eco.metrics["fallback_used"], 1)

    def test_call_unknown_with_fallback(self):
        self.eco.register_tool("b", lambda: "ok")
        result = self.eco.call("missing", {}, fallback_chain=["b"])
        self.assertEqual(result, "ok")

    def test_call_all_fail(self):
        self.eco.register_tool("a", lambda x: 1 / 0)
        result = self.eco.call("a", {"x": 1}, fallback_chain=["b"])
        self.assertIsNone(result)
        self.assertEqual(self.eco.metrics["errors"], 2)


class TestAudit(unittest.TestCase):
    def test_log_appended(self):
        eco = main.ToolEcosystem()
        eco.register_tool("foo", lambda x: x)
        eco.call("foo", {"x": 1})
        self.assertEqual(len(eco.audit_log), 1)


class TestMetrics(unittest.TestCase):
    def test_observe(self):
        eco = main.ToolEcosystem()
        m = eco.observe()
        self.assertIn("requests", m)
        self.assertIn("errors", m)


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