"""Pruebas para 20-benchmarks-webarena-osworld."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestWebArena(unittest.TestCase):
    def test_basic(self):
        preds = ["a", "b", "x"]
        truth = [
            {"site": "shopping", "answer": "a"},
            {"site": "reddit", "answer": "b"},
            {"site": "shopping", "answer": "c"},
        ]
        scores = main.webarena_score(preds, truth)
        # shopping: 1/2, reddit: 1/1
        self.assertEqual(scores["shopping"], 0.5)
        self.assertEqual(scores["reddit"], 1.0)


class TestOSWorld(unittest.TestCase):
    def test_basic(self):
        preds = ["a", "b"]
        truth = [
            {"app": "chrome", "answer": "a"},
            {"app": "vscode", "answer": "b"},
        ]
        scores = main.osworld_score(preds, truth)
        self.assertEqual(scores["chrome"], 1.0)
        self.assertEqual(scores["vscode"], 1.0)


class TestTaskCompletion(unittest.TestCase):
    def test_all(self):
        self.assertEqual(main.task_completion_rate(["a", "b"], [{"answer": "a"}, {"answer": "b"}]), 1.0)

    def test_none(self):
        self.assertEqual(main.task_completion_rate(["a", "b"], [{"answer": "c"}, {"answer": "d"}]), 0.0)

    def test_partial(self):
        # partial credit: 0.5 for partial matches
        truth = [{"answer": "a", "partial": True}, {"answer": "b", "partial": False}]
        preds = ["x", "b"]
        # first: 0.5 (partial credit, "x" not None), second: 1.0
        score = main.task_completion_rate(preds, truth)
        self.assertEqual(score, 0.75)

    def test_no_partial(self):
        self.assertEqual(main.task_completion_rate(["a", "b"], [{"answer": "c"}, {"answer": "d"}], partial_credit=False), 0.0)


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