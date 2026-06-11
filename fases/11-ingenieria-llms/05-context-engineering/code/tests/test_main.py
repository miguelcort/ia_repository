"""Pruebas para 05-context-engineering."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestBudget(unittest.TestCase):
    def test_under(self):
        ok, n = main.context_budget_check("hi", max_tokens=100)
        self.assertTrue(ok)
        self.assertEqual(n, 0)

    def test_over(self):
        ok, n = main.context_budget_check("x" * 1000, max_tokens=100)
        self.assertFalse(ok)


class TestPriority(unittest.TestCase):
    def test_priority(self):
        items = [
            {"content": "a" * 100, "priority": 1},
            {"content": "b" * 100, "priority": 10},
        ]
        out, used = main.context_priority(items, max_tokens=100)
        # b deberia estar primero
        self.assertEqual(out[0]["content"], "b" * 100)


class TestSliding(unittest.TestCase):
    def test_under(self):
        out = main.sliding_context("a b c", max_tokens=10)
        self.assertEqual(out, "a b c")

    def test_over(self):
        out = main.sliding_context("a b c d e f g h i j k", max_tokens=5)
        self.assertEqual(out, "g h i j k")


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