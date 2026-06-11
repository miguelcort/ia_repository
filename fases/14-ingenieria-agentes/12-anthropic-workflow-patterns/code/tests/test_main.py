"""Pruebas para 12-anthropic-workflow-patterns."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPromptChaining(unittest.TestCase):
    def test_basic(self):
        steps = ["a", "b", "c"]
        def llm(step, h):
            return f"out_{step}"
        result = main.prompt_chaining(steps, llm)
        self.assertEqual(result, "out_c")


class TestRouting(unittest.TestCase):
    def test_basic(self):
        classifier = lambda q: "x" if "x" in q else "y"
        handlers = {"x": lambda q: "x_result", "y": lambda q: "y_result"}
        route = main.routing(classifier, handlers)
        self.assertEqual(route("x query"), "x_result")
        self.assertEqual(route("y query"), "y_result")

    def test_no_match(self):
        classifier = lambda q: "z"
        handlers = {"x": lambda q: "x_result"}
        route = main.routing(classifier, handlers)
        self.assertIsNone(route("query"))


class TestParallelization(unittest.TestCase):
    def test_basic(self):
        tasks = ["a", "b"]
        results = main.parallelization(tasks, lambda t: f"r_{t}")
        self.assertEqual(results, ["r_a", "r_b"])


class TestOrchestratorWorkers(unittest.TestCase):
    def test_basic(self):
        def orch(q, n=None, results=None, mode=None):
            if mode is None:
                return [f"sub1_{q}", f"sub2_{q}"]
            return f"synth({results})"
        def worker(sub):
            return f"work_{sub}"
        out = main.orchestrator_workers("query", orch, worker, n_workers=2)
        self.assertIn("synth", out)


class TestEvaluatorOptimizer(unittest.TestCase):
    def test_converge(self):
        counter = [0]
        def eval_fn(o):
            counter[0] += 1
            if counter[0] >= 2:
                return 1.0
            return 0.0
        def opt_fn(o, s):
            return o + " x"
        out = main.evaluator_optimizer("draft", eval_fn, opt_fn, max_iterations=5)
        self.assertEqual(out["score"], 1.0)

    def test_max_iter(self):
        def eval_fn(o):
            return 0.0
        out = main.evaluator_optimizer("x", eval_fn, lambda o, s: o, max_iterations=2)
        self.assertEqual(out["score"], 0.0)


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