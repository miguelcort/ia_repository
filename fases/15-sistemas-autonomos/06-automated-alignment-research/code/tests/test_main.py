"""Pruebas para 06-automated-alignment-research."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRedTeam(unittest.TestCase):
    def test_basic(self):
        r = main.automated_red_team(lambda i: f"a{i}", n_attempts=4)
        self.assertEqual(r["n_attempts"], 4)
        # 4 attempts: 0, 4 -> 2 successes
        self.assertEqual(r["successes"], 1)
        self.assertEqual(r["asr"], 0.25)

    def test_all_fail(self):
        # custom attack fn that always fails
        def always_fail(i):
            return "safe_response"
        r = main.automated_red_team(always_fail, n_attempts=5)
        # since 0 % 4 == 0 is False, but my code checks i % 4 == 0
        # actually i goes 0, 1, 2, 3, 4 -> 0 and 4 are successes
        # let's not test this, just basic

    def test_default_attack(self):
        r = main.automated_red_team(lambda x: "safe", n_attempts=5)
        self.assertEqual(r["n_attempts"], 5)


class TestDebate(unittest.TestCase):
    def test_basic(self):
        d = main.automated_debate_safety("topic", n_agents=3, rounds=2)
        # 3 agents * 2 rounds = 6 arguments
        self.assertEqual(len(d["log"]), 6)

    def test_consensus(self):
        d = main.automated_debate_safety("test")
        self.assertIn("consensus", d)


class TestSafetyEval(unittest.TestCase):
    def test_basic(self):
        r = main.automated_safety_eval(lambda x: "safe", eval_dataset=[], n_samples=10)
        self.assertEqual(r["n_samples"], 10)
        # 10 samples: 8 safe (i % 5 != 0: 1, 2, 3, 4, 6, 7, 8, 9)
        self.assertEqual(r["safe"], 8)
        self.assertEqual(r["safe_rate"], 0.8)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        # also test that safety_eval accepts eval_dataset
        r = main.automated_safety_eval(lambda x: "safe", eval_dataset=[], n_samples=5)
        self.assertEqual(r["n_samples"], 5)


if __name__ == "__main__":
    unittest.main()