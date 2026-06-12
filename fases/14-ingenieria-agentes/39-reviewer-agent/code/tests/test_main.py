"""Pruebas para 39-reviewer-agent."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestChecklist(unittest.TestCase):
    def test_add_list(self):
        cl = main.ReviewChecklist()
        cl.add("a", lambda x: True)
        cl.add("b", lambda x: False)
        self.assertEqual(cl.list(), ["a", "b"])


class TestReviewer(unittest.TestCase):
    def test_all_pass(self):
        cl = main.ReviewChecklist()
        cl.add("a", lambda x: True)
        cl.add("b", lambda x: True)
        r = main.Reviewer(cl, threshold=0.5)
        result = r.review("anything")
        self.assertTrue(result["approved"])
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(len(result["passed"]), 2)
        self.assertEqual(len(result["failed"]), 0)

    def test_some_fail(self):
        cl = main.ReviewChecklist()
        cl.add("a", lambda x: True, weight=1.0)
        cl.add("b", lambda x: False, weight=2.0)
        r = main.Reviewer(cl, threshold=0.5)
        result = r.review("x")
        self.assertFalse(result["approved"])
        self.assertAlmostEqual(result["score"], 1/3, places=4)

    def test_all_fail(self):
        cl = main.ReviewChecklist()
        cl.add("a", lambda x: False)
        r = main.Reviewer(cl, threshold=0.5)
        result = r.review("x")
        self.assertFalse(result["approved"])
        self.assertEqual(result["score"], 0.0)

    def test_exception_in_check(self):
        cl = main.ReviewChecklist()
        cl.add("boom", lambda x: (_ for _ in ()).throw(ValueError("crash")))
        r = main.Reviewer(cl, threshold=0.5)
        result = r.review("x")
        self.assertFalse(result["approved"])
        self.assertEqual(len(result["failed"]), 1)

    def test_feedback(self):
        cl = main.ReviewChecklist()
        cl.add("a", lambda x: True)
        cl.add("b", lambda x: False)
        r = main.Reviewer(cl, threshold=0.5)
        result = r.review("x")
        self.assertIn("PASS: a", result["feedback"])
        self.assertIn("FAIL: b", result["feedback"])

    def test_empty_checklist(self):
        r = main.Reviewer(main.ReviewChecklist(), threshold=0.5)
        result = r.review("x")
        self.assertEqual(result["score"], 0)
        self.assertFalse(result["approved"])


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