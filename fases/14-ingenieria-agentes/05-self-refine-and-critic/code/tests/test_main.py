"""Pruebas para 05-self-refine-and-critic."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestCritiqueMemory(unittest.TestCase):
    def setUp(self):
        self.mem = main.CritiqueMemory(max_size=3)

    def test_add(self):
        self.mem.add("a")
        self.assertEqual(len(self.mem.get_all()), 1)

    def test_max_size(self):
        for i in range(5):
            self.mem.add(f"c{i}")
        self.assertEqual(len(self.mem.get_all()), 3)
        self.assertEqual(self.mem.get_all()[0]["content"], "c2")

    def test_latest(self):
        self.assertIsNone(self.mem.latest())
        self.mem.add("a")
        self.assertEqual(self.mem.latest()["content"], "a")


class TestGenerateCritique(unittest.TestCase):
    def test_no_critique(self):
        c = main.generate_critique("This is a good long output")
        self.assertIsNone(c)

    def test_short(self):
        c = main.generate_critique("hi")
        self.assertIn("too short", c)

    def test_error_keyword(self):
        c = main.generate_critique("this is an error message here")
        self.assertIn("error", c.lower())

    def test_missing_criteria(self):
        c = main.generate_critique("some output", criteria=["accuracy", "speed"])
        self.assertIn("missing", c)

    def test_all_criteria(self):
        c = main.generate_critique("accuracy and speed are good", criteria=["accuracy", "speed"])
        self.assertIsNone(c)


class TestSelfRefine(unittest.TestCase):
    def test_converge(self):
        def refine(out, crit):
            return "refined long output"
        result = main.self_refine("initial", refine, max_iterations=5)
        self.assertTrue(result["converged"])

    def test_max_iter(self):
        def refine(out, crit):
            return "error: " + out  # always contains error keyword
        result = main.self_refine("hi", refine, max_iterations=3)
        self.assertFalse(result["converged"])
        self.assertEqual(result["iterations"], 3)


class TestCRITIC(unittest.TestCase):
    def test_verify(self):
        def tool_fn(out):
            return {"correct": True}
        result = main.critic_with_tool_use("test", tool_fn, max_iterations=3)
        self.assertTrue(result["verified"])

    def test_correction(self):
        call_count = [0]
        def tool_fn(out):
            call_count[0] += 1
            if call_count[0] < 2:
                return {"correct": False, "correction": "fix"}
            return {"correct": True}
        result = main.critic_with_tool_use("test", tool_fn, max_iterations=3)
        self.assertTrue(result["verified"])


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