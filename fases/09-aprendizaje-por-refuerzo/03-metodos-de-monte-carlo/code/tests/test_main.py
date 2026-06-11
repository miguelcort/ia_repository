"""Pruebas para 03-metodos-de-monte-carlo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestFirstVisit(unittest.TestCase):
    def test_simple(self):
        # 1 episode, state 0 visto 1 vez
        states = [[0, 1, 2, 3]]
        returns = [4.0]
        V = main.first_visit_mc(states, returns, n_states=5)
        self.assertEqual(V[0], 4.0)
        # States no vistos deberian ser 0
        self.assertEqual(V[4], 0.0)

    def test_average(self):
        # 2 episodes con diferente return en state 0
        states = [[0, 1, 2], [0, 2, 3]]
        returns = [10.0, 6.0]
        V = main.first_visit_mc(states, returns, n_states=5)
        # V[0] = (10 + 6) / 2 = 8
        self.assertAlmostEqual(V[0], 8.0, places=4)


class TestEveryVisit(unittest.TestCase):
    def test_simple(self):
        # 1 episode, state 0 visto 2 veces
        states = [[0, 1, 0, 2]]
        returns = [5.0]
        V = main.every_visit_mc(states, returns, n_states=5)
        # V[0] deberia ser 5
        self.assertEqual(V[0], 5.0)


class TestSampleReturns(unittest.TestCase):
    def test_shape(self):
        returns = main.sample_returns(10, 4, np.array([1, 1, 1, 1]), 0.9, seed=0)
        self.assertEqual(len(returns), 10)


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