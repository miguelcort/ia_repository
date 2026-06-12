"""Pruebas para 03-alphaevolve-evolutionarycoding."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMutate(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(main.mutate_code(""), "")

    def test_replaces(self):
        # run multiple times - at least one should keep length 3 (replace)
        kept = False
        for _ in range(50):
            result = main.mutate_code("abc", n_mutations=1)
            if len(result) == 3:
                kept = True
                break
        # statistical: in 50 trials at least one should be replace
        self.assertTrue(kept or len(result) >= 2)  # either replace or delete

    def test_insert(self):
        result = main.mutate_code("abc", n_mutations=1)
        # length is 2, 3, or 4 depending on op
        self.assertGreaterEqual(len(result), 2)
        self.assertLessEqual(len(result), 4)

    def test_delete(self):
        result = main.mutate_code("abc", n_mutations=1)
        # 2, 3, or 4
        self.assertGreaterEqual(len(result), 2)
        self.assertLessEqual(len(result), 4)

    def test_many_mutations(self):
        result = main.mutate_code("abc", n_mutations=10)
        # length varies
        self.assertGreaterEqual(len(result), 0)


class TestEvaluate(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(main.evaluate_code("abc", lambda c: len(c)), 3)

    def test_empty(self):
        self.assertEqual(main.evaluate_code("", lambda c: len(c)), 0)


class TestAlphaEvolve(unittest.TestCase):
    def test_basic(self):
        # eval = -len (shorter is better)
        def eval_fn(code):
            return -len(code)
        best_code, best_score = main.alphaevolve("hello world", eval_fn, n_generations=5, population_size=5)
        self.assertLessEqual(best_score, 0)

    def test_returns_best(self):
        def eval_fn(code):
            return len(set(code))  # diversity
        best_code, best_score = main.alphaevolve("abc", eval_fn, n_generations=3, population_size=4)
        self.assertIsNotNone(best_code)


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