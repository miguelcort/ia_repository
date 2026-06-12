"""Pruebas para 19-ai-gateways."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestProvider(unittest.TestCase):
    def test_create(self):
        p = main.Provider("openai", "https://api.openai.com", models=["gpt-4o"])
        self.assertEqual(p.name, "openai")
        self.assertEqual(p.models, ["gpt-4o"])


class TestGateway(unittest.TestCase):
    def setUp(self):
        self.gw = main.AIGateway()
        self.gw.add_provider(main.Provider("openai", "https://api.openai.com", models=["gpt-4o"]))
        self.gw.add_provider(main.Provider("anthropic", "https://api.anthropic.com", models=["claude-3-5-sonnet"]))
        self.gw.add_route("smart", ["anthropic", "openai"])

    def test_add_provider(self):
        self.assertIn("openai", self.gw.providers)

    def test_add_route(self):
        self.assertIn("smart", self.gw.routes)

    def test_call_first(self):
        response, source = self.gw.call("smart", "claude-3-5-sonnet", "Hello")
        self.assertIn("anthropic", source)

    def test_call_fallback(self):
        response, source = self.gw.call("smart", "gpt-4o", "Hello")
        self.assertIn(source, ["anthropic", "openai"])

    def test_call_unknown_route(self):
        response, source = self.gw.call("unknown", "m", "x")
        self.assertIsNone(response)
        self.assertEqual(source, "no_provider")

    def test_cache(self):
        self.gw.call("smart", "claude-3-5-sonnet", "Cached")
        r1, s1 = self.gw.call("smart", "claude-3-5-sonnet", "Cached")
        self.assertEqual(s1, "cache")

    def test_list_routes(self):
        self.assertIn("smart", self.gw.list_routes())


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