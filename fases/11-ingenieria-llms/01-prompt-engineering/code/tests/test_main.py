"""Pruebas para 01-prompt-engineering."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestBuildPrompt(unittest.TestCase):
    def test_basic(self):
        p = main.build_prompt("Test instruction")
        self.assertIn("Test instruction", p)
        self.assertIn("### Response:", p)

    def test_with_system(self):
        p = main.build_prompt("Q", system="S")
        self.assertIn("### System:", p)
        self.assertIn("S", p)

    def test_with_examples(self):
        examples = [("a", "1"), ("b", "2")]
        p = main.build_prompt("Q", examples=examples)
        self.assertIn("### Example:", p)
        self.assertIn("Input: a", p)
        self.assertIn("Output: 1", p)


class TestCoT(unittest.TestCase):
    def test_cot_prompt(self):
        p = main.cot_prompt("Q?")
        self.assertIn("paso a paso", p)


class TestReAct(unittest.TestCase):
    def test_react(self):
        p = main.react_prompt("Q?", [("search", "busca")])
        self.assertIn("search:", p)
        self.assertIn("Thought:", p)


class TestTokensEstimate(unittest.TestCase):
    def test_estimate(self):
        # 100 chars / 4 = 25
        e = main.prompt_tokens_estimate("a" * 100)
        self.assertEqual(e, 25)


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