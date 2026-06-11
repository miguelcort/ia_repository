"""Pruebas para 11-planning-htn-and-evolutionary."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestHTN(unittest.TestCase):
    def test_primitive(self):
        methods = {}
        plan = main.htn_plan("walk", methods)
        self.assertEqual(plan, ["walk"])

    def test_one_level(self):
        methods = {"a": {"subtasks": ["b", "c"]}}
        plan = main.htn_plan("a", methods)
        self.assertEqual(plan, ["b", "c"])

    def test_nested(self):
        methods = {
            "a": {"subtasks": ["b", "c"]},
            "b": {"subtasks": ["d", "e"]},
        }
        plan = main.htn_plan("a", methods)
        self.assertEqual(plan, ["d", "e", "c"])

    def test_max_depth(self):
        methods = {"a": {"subtasks": ["b"]}, "b": {"subtasks": ["c"]}, "c": {"subtasks": ["d"]}}
        plan = main.htn_plan("a", methods, max_depth=1)
        # only one level of decomposition: a -> b, but b -> c not expanded
        self.assertEqual(plan, ["b"])


class TestEvolutionary(unittest.TestCase):
    def test_basic(self):
        initial = ["a", "b", "c"]
        # target: a, b, c (no penalty if correct)
        def eval_fn(p):
            score = 0
            for i, x in enumerate(p):
                if i < len(["a", "b", "c"]) and x == ["a", "b", "c"][i]:
                    score += 1
            return score
        def mutate(p):
            p = list(p)
            # random swap
            if len(p) > 1:
                i, j = 0, 1
                p[i], p[j] = p[j], p[i]
            return p
        best = main.evolutionary_plan(initial, eval_fn, mutate, n_generations=10, population_size=5)
        self.assertEqual(best, ["a", "b", "c"])


class TestLLMPlan(unittest.TestCase):
    def test_basic(self):
        result = main.llm_plan("test", lambda q: ["step1", "step2"])
        self.assertEqual(result, ["step1", "step2"])


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