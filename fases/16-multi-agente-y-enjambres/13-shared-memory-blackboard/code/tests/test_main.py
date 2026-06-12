"""Pruebas para 13-shared-memory-blackboard."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestBlackboard(unittest.TestCase):
    def setUp(self):
        self.bb = main.Blackboard()

    def test_write_read(self):
        self.bb.write("a", 1, "x")
        self.assertEqual(self.bb.read("a"), 1)

    def test_version(self):
        self.bb.write("a", 1, "x")
        self.assertEqual(self.bb.version("a"), 1)
        self.bb.write("a", 2, "y")
        self.assertEqual(self.bb.version("a"), 2)

    def test_lock(self):
        self.bb.lock("a", "x")
        self.assertTrue(self.bb.is_locked("a"))

    def test_lock_prevents_write(self):
        self.bb.lock("a", "x")
        with self.assertRaises(RuntimeError):
            self.bb.write("a", 1, "y")

    def test_unlock(self):
        self.bb.lock("a", "x")
        self.bb.unlock("a", "x")
        self.assertFalse(self.bb.is_locked("a"))
        self.bb.write("a", 1, "y")

    def test_double_lock_raises(self):
        self.bb.lock("a", "x")
        with self.assertRaises(RuntimeError):
            self.bb.lock("a", "y")

    def test_pattern_query(self):
        self.bb.write("task_a", 1, "x")
        self.bb.write("task_b", 2, "x")
        self.bb.write("user", 3, "x")
        result = self.bb.pattern_query("task_.*")
        self.assertEqual(len(result), 2)

    def test_access_log(self):
        self.bb.write("a", 1, "x")
        self.bb.write("a", 2, "y")
        log = self.bb.access_log("a")
        self.assertEqual(len(log), 2)

    def test_keys(self):
        self.bb.write("a", 1, "x")
        self.bb.write("b", 2, "y")
        self.assertEqual(set(self.bb.keys()), {"a", "b"})


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