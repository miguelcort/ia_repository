"""Pruebas para 10-rl-multi-agente."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestStochasticGame(unittest.TestCase):
    def test_step(self):
        n_states, n_actions = 3, 2
        P = np.zeros((n_states, n_actions, n_actions, n_states))
        P[0, 0, 0, 0] = 0.5
        P[0, 0, 0, 1] = 0.5
        R_a = np.zeros((n_states, n_actions, n_actions))
        R_b = np.zeros((n_states, n_actions, n_actions))
        s_next, ra, rb = main.stochastic_game_step(0, 0, 0, P, R_a, R_b, n_states)
        self.assertIn(s_next, [0, 1])


class TestNash(unittest.TestCase):
    def test_prisoner_dilemma(self):
        # Prisoner's dilemma: (defect, defect) = (1, 1) es Nash
        R_a = np.array([[3, 0], [5, 1]])
        R_b = np.array([[3, 5], [0, 1]])
        eq = main.nash_equilibrium_2p_payoff(R_a, R_b)
        # En prisoner's dilemma, (1, 1) = (defect, defect) es Nash
        self.assertTrue(any(a == 1 and b == 1 for a, b in eq))

    def test_coordination(self):
        # Coordination game: (0, 0) y (1, 1) son Nash
        R_a = np.array([[2, 0], [0, 2]])
        R_b = np.array([[2, 0], [0, 2]])
        eq = main.nash_equilibrium_2p_payoff(R_a, R_b)
        self.assertIn((0, 0), eq)
        self.assertIn((1, 1), eq)


class TestQMARL(unittest.TestCase):
    def test_shape(self):
        Q = np.zeros((3, 2, 2))
        Q = main.q_marl_update(Q, 0, 0, 1, 1.0, 0.0, 1, 0.1, 0.9, 2)
        self.assertEqual(Q.shape, (3, 2, 2))
        self.assertGreater(Q[0, 0, 1], 0)


class TestCoop(unittest.TestCase):
    def test_joint_action(self):
        # (a, b) -> joint action
        self.assertEqual(main.joint_action_to_idx(0, 0, 2), 0)
        self.assertEqual(main.joint_action_to_idx(0, 1, 2), 1)
        self.assertEqual(main.joint_action_to_idx(1, 0, 2), 2)
        self.assertEqual(main.joint_action_to_idx(1, 1, 2), 3)


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