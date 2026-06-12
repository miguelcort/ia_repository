"""Pruebas para 11-handoffs-and-routines."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestHandoff(unittest.TestCase):
    def test_create(self):
        h = main.Handoff("a1", "a2", "need", {"k": "v"})
        self.assertEqual(h.from_agent, "a1")
        self.assertEqual(h.to_agent, "a2")
        self.assertEqual(h.reason, "need")
        self.assertEqual(h.context["k"], "v")

    def test_id_unique(self):
        h1 = main.Handoff("a", "b", "")
        h2 = main.Handoff("a", "b", "")
        self.assertNotEqual(h1.id, h2.id)


class TestHandoffFn(unittest.TestCase):
    def test_basic(self):
        h = main.handoff("a1", "a2", "reason")
        self.assertEqual(h.from_agent, "a1")


class TestRoutine(unittest.TestCase):
    def test_create(self):
        r = main.Routine("test", ["a", "b"])
        self.assertEqual(r.name, "test")
        self.assertEqual(r.cursor, 0)

    def test_current(self):
        r = main.Routine("test", ["a", "b", "c"])
        self.assertEqual(r.current_step(), "a")

    def test_advance(self):
        r = main.Routine("test", ["a", "b"])
        r.advance("r1")
        self.assertEqual(r.cursor, 1)
        self.assertEqual(r.current_step(), "b")

    def test_is_done(self):
        r = main.Routine("test", ["a"])
        r.advance()
        self.assertTrue(r.is_done())
        self.assertIsNone(r.current_step())

    def test_run(self):
        r = main.Routine("test", ["a", "b", "c"])
        history = r.run(lambda s: f"res_{s}")
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]["result"], "res_a")


class TestHandoffToRoutine(unittest.TestCase):
    def test_basic(self):
        h = main.handoff_to_routine("a1", "a2", "onboard", "new user")
        self.assertEqual(h["type"], "handoff_with_routine")
        self.assertEqual(h["to"], "a2")


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