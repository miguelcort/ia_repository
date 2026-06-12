"""Pruebas para 02-star-family-reasoning."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestStar(unittest.TestCase):
    def test_generate(self):
        r = main.star_generate_rationale("What is 2+2?", lambda p: f"r({p[:10]})")
        self.assertIn("rationale", r)
        self.assertIn("answer", r)

    def test_filter(self):
        r = [{"answer": "4"}, {"answer": "wrong"}, {"answer": "5"}]
        correct = main.star_filter_correct(r, "4", lambda a, g: g in a)
        self.assertEqual(len(correct), 1)
        self.assertEqual(correct[0]["answer"], "4")

    def test_filter_all_wrong(self):
        r = [{"answer": "a"}, {"answer": "b"}]
        correct = main.star_filter_correct(r, "z", lambda a, g: g in a)
        self.assertEqual(correct, [])


class TestQuietStar(unittest.TestCase):
    def test_basic(self):
        r = main.quiet_star_reasoning("test", lambda p: f"r_{p[:8]}")
        self.assertIn("r_Think: ", r)

    def test_picks_longest(self):
        responses = ["short", "longer response", "medium"]
        r = main.quiet_star_reasoning("x", lambda p: responses.pop(0) if responses else "x")
        self.assertEqual(r, "longer response")


class TestVStar(unittest.TestCase):
    def test_majority(self):
        # mock con respuestas iguales
        r = main.v_star_self_consistency("q", lambda p: "same")
        self.assertEqual(r, "same")


class TestReST(unittest.TestCase):
    def test_basic(self):
        steps = [{"correct": True}, {"correct": False}, {"correct": True}]
        result = main.rest_train(steps, lambda d: d)
        self.assertEqual(result["trained_on"], 2)
        self.assertEqual(result["total"], 3)


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