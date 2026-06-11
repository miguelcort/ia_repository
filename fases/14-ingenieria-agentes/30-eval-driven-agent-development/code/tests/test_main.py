"""Pruebas para 30-eval-driven-agent-development."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestEvalSuite(unittest.TestCase):
    def setUp(self):
        self.suite = main.EvalSuite("test")

    def test_add(self):
        self.suite.add_case({"name": "t", "input": "x", "expected": "y"})
        self.assertEqual(len(self.suite.cases), 1)

    def test_run_all_pass(self):
        self.suite.add_case({"name": "t1", "input": "Hello", "expected": "hi"})
        def agent(x):
            return "hi!"
        summary = self.suite.run(agent)
        self.assertEqual(summary["score"], 1.0)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["total"], 1)

    def test_run_partial(self):
        self.suite.add_case({"name": "t1", "input": "Hello", "expected": "hi"})
        self.suite.add_case({"name": "t2", "input": "weather", "expected": "rain"})
        def agent(x):
            return "hi" if "Hello" in x else "sunny"
        summary = self.suite.run(agent)
        self.assertEqual(summary["score"], 0.5)

    def test_run_error(self):
        self.suite.add_case({"name": "t1", "input": "x", "expected": "y"})
        def agent(x):
            raise RuntimeError("boom")
        summary = self.suite.run(agent)
        self.assertEqual(summary["score"], 0.0)

    def test_run_empty(self):
        summary = self.suite.run(lambda x: "x")
        self.assertEqual(summary["score"], 0)

    def test_regression_detect(self):
        self.suite.add_case({"name": "t1", "input": "Hello", "expected": "hi"})
        def good(x):
            return "hi!"
        def bad(x):
            return "wrong"
        self.suite.run(good)
        self.suite.run(bad)
        self.assertTrue(self.suite.detect_regression())

    def test_no_regression(self):
        self.suite.add_case({"name": "t1", "input": "Hello", "expected": "hi"})
        self.suite.run(lambda x: "hi!")
        self.suite.run(lambda x: "hi!")
        self.assertFalse(self.suite.detect_regression())


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