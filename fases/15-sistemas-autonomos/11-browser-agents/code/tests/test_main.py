"""Pruebas para 11-browser-agents."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestBrowserAgents(unittest.TestCase):
    def test_list(self):
        agents = main.list_browser_agents()
        self.assertIn("browser_use", agents)
        self.assertIn("computer_use", agents)
        self.assertIn("operator", agents)
        self.assertIn("stagehand", agents)
        self.assertIn("skyvern", agents)

    def test_get(self):
        a = main.get_agent("computer_use")
        self.assertEqual(a["vendor"], "Anthropic")
        self.assertIn("screenshot", a["modalities"])

    def test_get_unknown(self):
        self.assertIsNone(main.get_agent("unknown"))

    def test_filter_oss(self):
        oss = main.filter_open_source()
        self.assertIn("browser_use", oss)
        self.assertIn("stagehand", oss)
        self.assertIn("skyvern", oss)
        self.assertNotIn("computer_use", oss)
        self.assertNotIn("operator", oss)

    def test_filter_workflows(self):
        wf = main.filter_with_workflows()
        self.assertIn("skyvern", wf)

    def test_computer_use_claude(self):
        a = main.get_agent("computer_use")
        self.assertIn("claude", a["claude_model"])

    def test_operator_gpt(self):
        a = main.get_agent("operator")
        self.assertIn("computer", a["gpt_model"])


class TestPlanBrowserTask(unittest.TestCase):
    def test_basic_plan(self):
        plan = main.plan_browser_task({
            "url": "https://example.com",
            "fields": ["title", "price"],
            "submit": False,
        })
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["step"], "navigate")
        self.assertEqual(plan[1]["step"], "extract")

    def test_with_submit(self):
        plan = main.plan_browser_task({
            "url": "https://example.com",
            "fields": ["email"],
            "submit": True,
        })
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[2]["step"], "submit")

    def test_with_extra_steps(self):
        plan = main.plan_browser_task(
            {"url": "https://x.com", "fields": ["a"]},
            steps=[{"step": "login"}],
        )
        self.assertEqual(plan[0]["step"], "login")
        self.assertEqual(plan[1]["step"], "navigate")


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