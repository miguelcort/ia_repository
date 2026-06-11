"""Pruebas para 04-q-learning-y-sarsa."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestQLearning(unittest.TestCase):
    def test_update_shape(self):
        Q = np.zeros((3, 2))
        Q = main.q_learning_update(Q, 0, 0, 1.0, 1, alpha=0.1, gamma=0.9, n_actions=2)
        self.assertEqual(Q.shape, (3, 2))
        # Q[0, 0] deberia ser > 0 (target 1 + 0)
        self.assertGreater(Q[0, 0], 0)

    def test_converge(self):
        n_states, n_actions = 3, 2
        Q = np.zeros((n_states, n_actions))
        # Mock: s=0 -> s=1 con reward 1, V(s=1)=0
        for _ in range(100):
            Q = main.q_learning_update(Q, 0, 0, 1.0, 1, 0.5, 0.9, n_actions)
        # Q[0, 0] deberia converger a 1.0 (1 reward + 0.9 * 0)
        self.assertAlmostEqual(Q[0, 0], 1.0, places=3)


class TestSARSA(unittest.TestCase):
    def test_update(self):
        Q = np.zeros((3, 2))
        Q = main.sarsa_update(Q, 0, 0, 1.0, 1, 1, 0.1, 0.9)
        self.assertEqual(Q.shape, (3, 2))
        self.assertGreater(Q[0, 0], 0)

    def test_on_policy_uses_actual_action(self):
        Q = np.zeros((3, 2))
        Q[1, 0] = 5.0  # Q(s', 0) alto
        Q[1, 1] = 0.0  # Q(s', 1) bajo
        # SARSA con a_next=0: usa 5.0
        Q1 = main.sarsa_update(Q.copy(), 0, 1, 1.0, 1, 0, 0.1, 0.9)
        # SARSA con a_next=1: usa 0.0
        Q2 = main.sarsa_update(Q.copy(), 0, 1, 1.0, 1, 1, 0.1, 0.9)
        # Q1 deberia ser mayor (target mas alto)
        self.assertGreater(Q1[0, 1], Q2[0, 1])


class TestEpsilonGreedy(unittest.TestCase):
    def test_argmax(self):
        Q = np.array([[0, 5, 0]])
        # Sin rng deterministico, deberia ser argmax = 1
        a = main.epsilon_greedy(Q, 0, 3, epsilon=0.0)
        self.assertEqual(a, 1)


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