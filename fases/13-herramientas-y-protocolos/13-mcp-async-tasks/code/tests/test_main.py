"""Pruebas para 13-mcp-async-tasks."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestTask(unittest.TestCase):
    def test_create(self):
        t = main.Task("t1", "op", {"a": 1})
        self.assertEqual(t.id, "t1")
        self.assertEqual(t.status, "pending")
        self.assertEqual(t.params, {"a": 1})


class TestTaskManager(unittest.TestCase):
    def setUp(self):
        self.tm = main.TaskManager()

    def test_create_task(self):
        t = self.tm.create_task("op")
        self.assertIsNotNone(t)
        self.assertIn(t.id, self.tm.tasks)

    def test_unique_ids(self):
        t1 = self.tm.create_task("op")
        t2 = self.tm.create_task("op")
        self.assertNotEqual(t1.id, t2.id)

    def test_get_task(self):
        t = self.tm.create_task("op")
        got = self.tm.get_task(t.id)
        self.assertEqual(got.id, t.id)

    def test_get_unknown(self):
        self.assertIsNone(self.tm.get_task("nope"))

    def test_list_tasks(self):
        self.tm.create_task("a")
        self.tm.create_task("b")
        tasks = self.tm.list_tasks()
        self.assertEqual(len(tasks), 2)

    def test_list_by_status(self):
        t1 = self.tm.create_task("a")
        self.tm.mark_running(t1.id)
        running = self.tm.list_tasks(status="running")
        self.assertEqual(len(running), 1)

    def test_cancel(self):
        t = self.tm.create_task("op")
        ok = self.tm.cancel_task(t.id)
        self.assertTrue(ok)
        self.assertEqual(t.status, "cancelled")

    def test_cancel_completed(self):
        t = self.tm.create_task("op")
        self.tm.mark_completed(t.id, "x")
        ok = self.tm.cancel_task(t.id)
        self.assertFalse(ok)

    def test_mark_running_completed(self):
        t = self.tm.create_task("op")
        self.tm.mark_running(t.id)
        self.tm.mark_completed(t.id, {"out": 1})
        self.assertEqual(t.status, "completed")
        self.assertEqual(t.result, {"out": 1})

    def test_mark_failed(self):
        t = self.tm.create_task("op")
        self.tm.mark_failed(t.id, "boom")
        self.assertEqual(t.status, "failed")
        self.assertEqual(t.error, "boom")


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