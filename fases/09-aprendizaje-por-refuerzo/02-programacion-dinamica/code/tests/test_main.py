"""Pruebas para 02-programacion-dinamica."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestValueIteration(unittest.TestCase):
    def test_converge(self):
        n_s, n_a = 3, 2
        P = np.zeros((n_s, n_a, n_s))
        P[0, 0, 0] = 0.5; P[0, 0, 1] = 0.5
        P[0, 1, 1] = 1.0
        P[1, 0, 0] = 0.5; P[1, 0, 2] = 0.5
        P[1, 1, 2] = 1.0
        P[2, :, 2] = 1.0
        R = np.array([[0, 0], [1, 1], [0, 0]])
        V, pi = main.value_iteration(P, R, gamma=0.9, n_states=n_s, n_actions=n_a)
        # V deberia converger
        self.assertEqual(V.shape, (n_s,))
        self.assertEqual(pi.shape, (n_s,))

    def test_action_in_range(self):
        n_s, n_a = 4, 3
        P = np.zeros((n_s, n_a, n_s))
        for s in range(n_s):
            for a in range(n_a):
                P[s, a, (s + 1) % n_s] = 1.0
        R = np.zeros((n_s, n_a))
        R[0, :] = 1.0
        V, pi = main.value_iteration(P, R, gamma=0.9, n_states=n_s, n_actions=n_a)
        # policy deberia ser 0, 1, o 2
        for s in range(n_s):
            self.assertIn(pi[s], [0, 1, 2])


class TestPolicyIteration(unittest.TestCase):
    def test_converge(self):
        n_s, n_a = 3, 2
        P = np.zeros((n_s, n_a, n_s))
        P[0, 0, 0] = 0.5; P[0, 0, 1] = 0.5
        P[0, 1, 1] = 1.0
        P[1, 0, 0] = 0.5; P[1, 0, 2] = 0.5
        P[1, 1, 2] = 1.0
        P[2, :, 2] = 1.0
        R = np.array([[0, 0], [1, 1], [0, 0]])
        V, pi = main.policy_iteration(P, R, gamma=0.9, n_states=n_s, n_actions=n_a)
        self.assertEqual(V.shape, (n_s,))
        self.assertEqual(pi.shape, (n_s,))


class TestGridworld(unittest.TestCase):
    def test_gridworld(self):
        P, R, n_s, n_a = main.gridworld_value(4)
        self.assertEqual(n_s, 16)
        self.assertEqual(n_a, 4)

    def test_gridworld_solve(self):
        P, R, n_s, n_a = main.gridworld_value(4)
        V, pi = main.value_iteration(P, R, gamma=0.99, n_states=n_s, n_actions=n_a)
        # V deberia ser > 0 cerca del goal
        # Goal esta en state 3 (row 0, col 3)
        # V[3] deberia ser max
        self.assertGreater(V[3], V[0])


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