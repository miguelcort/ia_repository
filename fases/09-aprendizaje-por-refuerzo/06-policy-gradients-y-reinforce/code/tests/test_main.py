"""Pruebas para 06-policy-gradients-y-reinforce."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPolicyForward(unittest.TestCase):
    def test_shape(self):
        state = np.array([1.0, 0, 0, 0])
        W = np.random.default_rng(0).standard_normal((4, 2)) * 0.1
        b = np.zeros(2)
        probs = main.policy_forward(state, W, b, n_actions=2)
        self.assertEqual(probs.shape, (2,))

    def test_suma_uno(self):
        state = np.array([1.0, 0, 0, 0])
        W = np.random.default_rng(0).standard_normal((4, 2)) * 0.1
        b = np.zeros(2)
        probs = main.policy_forward(state, W, b, n_actions=2)
        self.assertAlmostEqual(probs.sum(), 1.0, places=6)


class TestReturns(unittest.TestCase):
    def test_returns(self):
        rewards = [0, 0, 1]
        gamma = 0.9
        G = main.discounted_returns(rewards, gamma)
        # G[2] = 1, G[1] = 0.9, G[0] = 0.81
        np.testing.assert_array_almost_equal(G, [0.81, 0.9, 1.0], decimal=4)

    def test_all_zero(self):
        G = main.discounted_returns([0, 0, 0], 0.9)
        np.testing.assert_array_equal(G, [0, 0, 0])


class TestReinforce(unittest.TestCase):
    def test_update_shape(self):
        n_states, n_actions = 4, 2
        W = np.zeros((n_states, n_actions))
        b = np.zeros(n_actions)
        states = [np.array([1, 0, 0, 0], dtype=float),
                  np.array([0, 1, 0, 0], dtype=float)]
        actions = [0, 1]
        rewards = [1.0, 0.0]
        W, b = main.reinforce_episode(states, actions, rewards, W, b, gamma=0.9, lr=0.1)
        # W deberia haber cambiado
        self.assertGreater(np.abs(W).max(), 0)

    def test_favors_high_return(self):
        n_states, n_actions = 4, 2
        W = np.zeros((n_states, n_actions))
        b = np.zeros(n_actions)
        # Estado 0, action 0 -> reward 1
        # Estado 0, action 1 -> reward 0
        states = [np.array([1, 0, 0, 0], dtype=float)] * 10
        actions = [0] * 10
        rewards = [1.0] * 10
        W, b = main.reinforce_episode(states, actions, rewards, W, b, gamma=0.9, lr=0.5)
        # W[0, 0] deberia ser positivo (prob de action 0 aumento)
        self.assertGreater(W[0, 0], 0)


class TestBaseline(unittest.TestCase):
    def test_baseline_subtract(self):
        r = np.array([1.0, 2.0, 3.0])
        b = 2.0
        adj = main.baseline_returns(r, b)
        np.testing.assert_array_equal(adj, [-1.0, 0.0, 1.0])


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