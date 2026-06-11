"""Pruebas para 09-constitutional-ai-y-mejora-self."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPrinciples(unittest.TestCase):
    def test_seis(self):
        p = main.constitutional_principles()
        self.assertEqual(len(p), 6)


class TestSteps(unittest.TestCase):
    def test_seis(self):
        s = main.rlaif_steps()
        self.assertEqual(len(s), 6)


class TestSelfCritique(unittest.TestCase):
    def test_score_range(self):
        score = main.self_critique_score("response", ["p1", "p2"])
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


class TestComparison(unittest.TestCase):
    def test_keys(self):
        c = main.constitutional_vs_rlhf()
        self.assertIn("RLHF", c)
        self.assertIn("RLAIF", c)
        self.assertIn("Constitutional AI", c)


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