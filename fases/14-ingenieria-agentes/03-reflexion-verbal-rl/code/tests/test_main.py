"""Pruebas para 03-reflexion-verbal-rl."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.mem = main.ReflexionMemory(max_size=3)

    def test_add(self):
        self.mem.add("reflection 1")
        self.assertEqual(len(self.mem.get_all()), 1)

    def test_max_size(self):
        for i in range(5):
            self.mem.add(f"reflection {i}")
        # max=3 -> keeps last 3
        self.assertEqual(len(self.mem.get_all()), 3)
        self.assertEqual(self.mem.get_all()[0]["content"], "reflection 2")

    def test_format(self):
        self.mem.add("a")
        self.mem.add("b")
        f = self.mem.format_for_prompt()
        self.assertIn("a", f)
        self.assertIn("b", f)

    def test_empty_format(self):
        f = self.mem.format_for_prompt()
        self.assertIn("No prior", f)

    def test_last(self):
        self.assertIsNone(self.mem.last())
        self.mem.add("a")
        self.assertEqual(self.mem.last()["content"], "a")


class TestReflection(unittest.TestCase):
    def test_error(self):
        r = main.generate_reflection({}, error="boom")
        self.assertIn("boom", r)
        self.assertIn("Mistake", r)

    def test_failure(self):
        r = main.generate_reflection({"success": False, "reason": "wrong path"})
        self.assertIn("Failed", r)
        self.assertIn("wrong path", r)

    def test_success(self):
        r = main.generate_reflection({"success": True, "strategy": "x"})
        self.assertIn("Succeeded", r)
        self.assertIn("x", r)


class TestReflexionLoop(unittest.TestCase):
    def test_success_first_try(self):
        def trial(q, mem):
            return {"success": True, "result": "ok"}
        mem = main.ReflexionMemory()
        out = main.reflexion_loop("test", trial, mem, max_trials=3)
        self.assertTrue(out["success"])
        self.assertEqual(out["trials"], 1)

    def test_success_after_failure(self):
        call_count = [0]
        def trial(q, mem):
            call_count[0] += 1
            if call_count[0] < 2:
                return {"success": False, "reason": "missing"}
            return {"success": True, "result": "ok"}
        mem = main.ReflexionMemory()
        out = main.reflexion_loop("test", trial, mem, max_trials=3)
        self.assertTrue(out["success"])
        self.assertEqual(out["trials"], 2)

    def test_max_trials_reached(self):
        def trial(q, mem):
            return {"success": False, "reason": "nope"}
        mem = main.ReflexionMemory()
        out = main.reflexion_loop("test", trial, mem, max_trials=3)
        self.assertFalse(out["success"])
        self.assertEqual(out["trials"], 3)


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