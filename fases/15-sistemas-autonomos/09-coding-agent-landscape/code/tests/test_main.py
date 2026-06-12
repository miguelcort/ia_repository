"""Pruebas para 09-coding-agent-landscape."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestCodingAgents(unittest.TestCase):
    def test_list(self):
        agents = main.list_coding_agents()
        self.assertIn("devin", agents)
        self.assertIn("cursor", agents)
        self.assertIn("claude_code", agents)
        self.assertIn("aider", agents)

    def test_get(self):
        a = main.get_agent("devin")
        self.assertEqual(a["vendor"], "Cognition")
        self.assertEqual(a["open_source"], False)

    def test_get_unknown(self):
        self.assertIsNone(main.get_agent("unknown"))

    def test_filter_open_source(self):
        open_source = main.filter_agents({"open_source": True})
        slugs = [s for s, _ in open_source]
        self.assertIn("aider", slugs)
        self.assertIn("cline", slugs)
        self.assertNotIn("devin", slugs)

    def test_filter_2024(self):
        agents_2024 = main.filter_agents({"release_year": 2024})
        self.assertGreater(len(agents_2024), 0)

    def test_filter_no_match(self):
        no_match = main.filter_agents({"vendor": "unknown"})
        self.assertEqual(no_match, [])

    def test_devin_has_tools(self):
        a = main.get_agent("devin")
        self.assertIn("shell", a["tools"])
        self.assertIn("browser", a["tools"])


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