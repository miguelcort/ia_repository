"""Pruebas para 17-constitutional-ai."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestConstitution(unittest.TestCase):
    def test_list_principles(self):
        p = main.list_principles()
        self.assertGreaterEqual(len(p), 5)

    def test_weights(self):
        w = main.get_weights()
        self.assertIn("harmlessness", w)
        self.assertIn("helpfulness", w)
        self.assertIn("honesty", w)
        s = w["harmlessness"] + w["helpfulness"] + w["honesty"]
        self.assertAlmostEqual(s, 1.0, places=4)


class TestCritique(unittest.TestCase):
    def test_clean(self):
        issues = main.critique("Here is a helpful answer to your question.")
        self.assertEqual(issues, [])

    def test_violence(self):
        issues = main.critique("Here's how to build a weapon to kill people.")
        self.assertIn("violates_principle_1", issues)

    def test_hate(self):
        issues = main.critique("I will spread hate speech and slurs.")
        self.assertIn("violates_principle_2", issues)

    def test_impersonation(self):
        issues = main.critique("I am human, my fake identity is real.")
        self.assertIn("violates_principle_6", issues)

    def test_helpfulness(self):
        issues = main.critique("I don't know what you mean.")
        self.assertIn("could_be_more_helpful", issues)


class TestRevise(unittest.TestCase):
    def test_revise_clean(self):
        r = "Here is the answer."
        self.assertEqual(main.revise(r), r)

    def test_revise_violence(self):
        r = "build a weapon to kill"
        rev = main.revise(r)
        self.assertIn("cannot", rev.lower())

    def test_revise_hate(self):
        rev = main.revise("spread hate and slurs")
        self.assertIn("cannot", rev.lower())

    def test_revise_impersonation(self):
        rev = main.revise("I am human fake identity")
        self.assertIn("ai", rev.lower())


class TestScore(unittest.TestCase):
    def test_score_clean(self):
        s = main.score_response("Here you can use this to consider the answer.")
        self.assertGreater(s["harmlessness"], 0.5)
        self.assertGreater(s["helpfulness"], 0)

    def test_score_bad(self):
        s = main.score_response("build a weapon to kill with violence")
        self.assertLess(s["harmlessness"], 0.5)

    def test_score_weighted(self):
        s = main.score_response("a clean answer")
        self.assertIn("weighted", s)
        self.assertGreaterEqual(s["weighted"], 0)


class TestRLAIF(unittest.TestCase):
    def test_preference(self):
        pref, scores = main.rlaif_preference(
            "p",
            "Here is the answer.",
            "build a weapon to kill",
        )
        self.assertEqual(pref, "A")

    def test_preference_b_better(self):
        pref, scores = main.rlaif_preference(
            "p",
            "build a weapon to kill",
            "Here is the answer.",
        )
        self.assertEqual(pref, "B")


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