"""Pruebas para 40-multi-session-handoff."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSession(unittest.TestCase):
    def test_create(self):
        s = main.Session()
        self.assertIsNotNone(s.session_id)
        self.assertEqual(s.context, {})

    def test_update_context(self):
        s = main.Session()
        s.update("a", 1)
        s.update("b", 2)
        self.assertEqual(s.context["a"], 1)
        self.assertEqual(s.context["b"], 2)

    def test_append(self):
        s = main.Session()
        s.append("user", "hi")
        s.append("assistant", "hello")
        self.assertEqual(len(s.transcript), 2)

    def test_snapshot(self):
        s = main.Session()
        s.update("a", 1)
        s.append("user", "x")
        snap = s.snapshot()
        self.assertEqual(snap["context"]["a"], 1)
        self.assertEqual(len(snap["transcript"]), 1)

    def test_from_snapshot(self):
        s1 = main.Session()
        s1.update("a", 1)
        snap = s1.snapshot()
        s2 = main.Session.from_snapshot(snap)
        self.assertEqual(s2.context["a"], 1)
        self.assertEqual(s2.session_id, s1.session_id)


class TestHandoffRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = main.HandoffRegistry()

    def test_create_get(self):
        s = self.reg.create()
        self.assertIs(self.reg.get(s.session_id), s)

    def test_get_unknown(self):
        self.assertIsNone(self.reg.get("nope"))

    def test_handoff_full(self):
        s1 = self.reg.create({"a": 1, "b": 2})
        s2 = self.reg.handoff(s1.session_id)
        self.assertEqual(s2.context, {"a": 1, "b": 2})
        self.assertNotEqual(s1.session_id, s2.session_id)

    def test_handoff_subset(self):
        s1 = self.reg.create({"a": 1, "b": 2})
        s2 = self.reg.handoff(s1.session_id, context_subset=["a"])
        self.assertEqual(s2.context, {"a": 1})

    def test_handoff_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.reg.handoff("nope")

    def test_list(self):
        self.reg.create()
        self.reg.create()
        self.assertEqual(len(self.reg.list()), 2)

    def test_delete(self):
        s = self.reg.create()
        self.assertTrue(self.reg.delete(s.session_id))
        self.assertIsNone(self.reg.get(s.session_id))
        self.assertFalse(self.reg.delete(s.session_id))


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