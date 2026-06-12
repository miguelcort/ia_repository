"""Pruebas para 23-failure-modes-mast-groupthink."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMAST(unittest.TestCase):
    def test_categories(self):
        cats = list(main.MAST_CATEGORIES.keys())
        self.assertIn("verification", cats)
        self.assertIn("multi_agent", cats)


class TestGroupthink(unittest.TestCase):
    def test_groupthink(self):
        self.assertTrue(main.detect_groupthink([], ["a", "a", "a", "a"]))

    def test_no_groupthink_diverse(self):
        self.assertFalse(main.detect_groupthink([], ["a", "b", "c", "d"]))

    def test_no_groupthink_low(self):
        self.assertFalse(main.detect_groupthink([], ["a", "a", "b", "b"]))

    def test_empty(self):
        self.assertFalse(main.detect_groupthink([], []))


class TestFreeRiding(unittest.TestCase):
    def test_free_rider(self):
        self.assertTrue(main.detect_free_riding({"a": 10, "b": 0.5}))

    def test_no_free_rider(self):
        self.assertFalse(main.detect_free_riding({"a": 10, "b": 9}))

    def test_empty(self):
        self.assertFalse(main.detect_free_riding({}))


class TestCascade(unittest.TestCase):
    def test_cascade(self):
        self.assertTrue(main.detect_cascade(["a", "a", "a", "a"], "a"))

    def test_no_cascade(self):
        self.assertFalse(main.detect_cascade(["a", "b", "c", "d"], "a"))

    def test_empty(self):
        self.assertFalse(main.detect_cascade([], "a"))


class TestDeadlock(unittest.TestCase):
    def test_deadlock(self):
        self.assertTrue(main.detect_deadlock({"a1": {"wait_steps": 10}}))

    def test_no_deadlock(self):
        self.assertFalse(main.detect_deadlock({"a1": {"wait_steps": 1}}))

    def test_empty(self):
        self.assertFalse(main.detect_deadlock({}))


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