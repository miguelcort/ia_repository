"""Pruebas para 01-long-horizon-agents."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestLongHorizonTask(unittest.TestCase):
    def test_step(self):
        t = main.LongHorizonTask("test", total_steps=3)
        r = t.step("a")
        self.assertEqual(r["step"], 1)
        self.assertEqual(t.completed, 1)

    def test_done(self):
        t = main.LongHorizonTask("test", total_steps=2)
        t.step("a")
        t.step("a")
        r = t.step("a")
        self.assertEqual(r["status"], "done")

    def test_checkpoint(self):
        t = main.LongHorizonTask("test", total_steps=5)
        t.step("a")
        t.step("b")
        cp = t.checkpoint()
        self.assertEqual(cp["completed"], 2)
        self.assertIn("step_1", cp["state"])

    def test_restore(self):
        t = main.LongHorizonTask("test", total_steps=5)
        t.step("a")
        t.step("b")
        cp = t.checkpoint()
        t2 = main.LongHorizonTask("test", total_steps=5)
        t2.restore(cp)
        self.assertEqual(t2.completed, 2)
        self.assertIn("step_1", t2.state)

    def test_is_done(self):
        t = main.LongHorizonTask("test", total_steps=2)
        t.step("a")
        t.step("a")
        self.assertTrue(t.is_done())


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