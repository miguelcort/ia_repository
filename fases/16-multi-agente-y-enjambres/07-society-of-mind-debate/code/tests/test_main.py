"""Pruebas para 07-society-of-mind-debate."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestDebateAgent(unittest.TestCase):
    def test_argue(self):
        a = main.DebateAgent("a1", "optimist", lambda t, prior=None: f"Yes {t}")
        result = a.argue("ship")
        self.assertEqual(result["perspective"], "optimist")
        self.assertIn("Yes", result["stance"])


class TestDebate(unittest.TestCase):
    def test_single_round(self):
        agents = [
            main.DebateAgent("a1", "p1", lambda t, prior=None: f"s1 {t}"),
            main.DebateAgent("a2", "p2", lambda t, prior=None: f"s2 {t}"),
        ]
        stmts = main.debate("topic", agents, rounds=1)
        self.assertEqual(len(stmts), 2)

    def test_multi_round(self):
        agents = [
            main.DebateAgent("a1", "p1", lambda t, prior=None: f"s1 {t}"),
        ]
        stmts = main.debate("topic", agents, rounds=3)
        self.assertEqual(len(stmts), 3)

    def test_passes_prior(self):
        seen_prior = []
        def fn(t, prior=None):
            seen_prior.extend(prior or [])
            return "x"
        agents = [main.DebateAgent("a1", "p1", fn)]
        main.debate("topic", agents, rounds=2)


class TestMajorityVote(unittest.TestCase):
    def test_majority(self):
        stmts = [
            {"stance": "yes", "perspective": "p1"},
            {"stance": "yes", "perspective": "p2"},
            {"stance": "no", "perspective": "p3"},
        ]
        self.assertEqual(main.majority_vote(stmts), "yes")

    def test_tie(self):
        stmts = [
            {"stance": "yes"},
            {"stance": "no"},
        ]
        result = main.majority_vote(stmts)
        self.assertIn(result, ["yes", "no"])

    def test_empty(self):
        self.assertIsNone(main.majority_vote([]))


class TestSynthesize(unittest.TestCase):
    def test_synth(self):
        stmts = [
            {"stance": "yes", "perspective": "opt"},
            {"stance": "no", "perspective": "pess"},
            {"stance": "yes", "perspective": "other"},
        ]
        s = main.synthesize(stmts)
        self.assertIn("opt", s)
        self.assertIn("pess", s)


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