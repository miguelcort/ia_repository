"""Pruebas para 25-multimodal-agents-computer-use."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEncodeScreen(unittest.TestCase):
    def test_shape(self):
        img = np.random.default_rng(0).standard_normal((1080, 1920, 3))
        feats = main.encode_screenshot(img, embed_dim=1024)
        # 1080/14 = 77, 1920/14 = 137
        n_h = 1080 // 14
        n_w = 1920 // 14
        self.assertEqual(feats.shape, (n_h * n_w + 1, 1024))


class TestClickCoords(unittest.TestCase):
    def test_range(self):
        img = np.random.default_rng(0).standard_normal((100, 100, 3))
        x, y = main.predict_click_coordinates(img, "button", screen_size=(1920, 1080))
        self.assertGreaterEqual(x, 0)
        self.assertLess(x, 1920)
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, 1080)


class TestPredictAction(unittest.TestCase):
    def test_basic(self):
        img = np.random.default_rng(0).standard_normal((100, 100, 3))
        action = main.predict_action(img, "click submit")
        self.assertIn(action["action"], ("click", "type", "scroll", "back"))


class TestExecute(unittest.TestCase):
    def test_execute(self):
        result = main.execute_action({"action": "click", "x": 100, "y": 200})
        self.assertIn("click", result)


class TestPlanning(unittest.TestCase):
    def test_loop(self):
        img = np.random.default_rng(0).standard_normal((100, 100, 3))
        history = main.planning_loop("login", img, max_steps=3)
        self.assertLessEqual(len(history), 3)
        self.assertGreater(len(history), 0)


class TestOSAtlas(unittest.TestCase):
    def test_grounding(self):
        img = np.random.default_rng(0).standard_normal((100, 100, 3))
        out = main.os_atlas_ui_grounding(img, "find search box")
        self.assertIn("x", out)
        self.assertIn("y", out)


class TestShowUI(unittest.TestCase):
    def test_basic(self):
        img = np.random.default_rng(0).standard_normal((100, 100, 3))
        action = main.showui_action(img, "scroll down")
        self.assertIn("action", action)


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