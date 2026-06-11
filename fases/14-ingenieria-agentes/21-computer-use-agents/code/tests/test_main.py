"""Pruebas para 21-computer-use-agents."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestScreen(unittest.TestCase):
    def test_add_find(self):
        s = main.Screen()
        s.add_element(100, 100, 200, 50, "button", "Login")
        coords = s.find_element("Login")
        self.assertEqual(coords, (200, 125))  # center

    def test_find_not_found(self):
        s = main.Screen()
        self.assertIsNone(s.find_element("missing"))


class TestComputerUseAgent(unittest.TestCase):
    def setUp(self):
        self.screen = main.Screen()
        self.screen.add_element(100, 100, 200, 50, "button", "Login")
        self.agent = main.ComputerUseAgent("test")

    def test_observe(self):
        obs = self.agent.observe(self.screen)
        self.assertIn("mock", obs)

    def test_predict_click(self):
        action = self.agent.predict_action("obs", "click Submit")
        self.assertEqual(action["action"], "click")

    def test_predict_type(self):
        action = self.agent.predict_action("obs", "type hello world")
        self.assertEqual(action["action"], "type")

    def test_predict_done(self):
        action = self.agent.predict_action("obs", "do nothing")
        self.assertEqual(action["action"], "done")

    def test_execute_click(self):
        action = {"action": "click", "target": "Login"}
        result = self.agent.execute(action, self.screen)
        self.assertIn("clicked", result)

    def test_execute_click_not_found(self):
        action = {"action": "click", "target": "missing"}
        result = self.agent.execute(action, self.screen)
        self.assertIn("not found", result)

    def test_execute_type(self):
        action = {"action": "type", "text": "hello"}
        result = self.agent.execute(action, self.screen)
        self.assertIn("typed", result)

    def test_run_loop(self):
        history = self.agent.run(self.screen, "click Login")
        self.assertGreater(len(history), 0)
        # the click action should have coords
        click_action = next((h for h in history if h["action"]["action"] == "click"), None)
        self.assertIsNotNone(click_action)


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