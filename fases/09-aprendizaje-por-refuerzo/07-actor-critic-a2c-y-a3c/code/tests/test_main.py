"""Pruebas para 07-actor-critic-a2c-y-a3c."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestActorCritic(unittest.TestCase):
    def test_init(self):
        ac = main.ActorCritic(n_states=4, n_actions=2, hidden=8, seed=0)
        self.assertEqual(ac.n_states, 4)
        self.assertEqual(ac.n_actions, 2)

    def test_forward(self):
        ac = main.ActorCritic(n_states=4, n_actions=2, hidden=8, seed=0)
        s = np.array([1, 0, 0, 0], dtype=float)
        probs, value = ac.forward(s)
        self.assertEqual(probs.shape, (2,))
        self.assertAlmostEqual(probs.sum(), 1.0, places=6)
        self.assertIsInstance(float(value), float)

    def test_select_action(self):
        ac = main.ActorCritic(n_states=4, n_actions=2, hidden=8, seed=0)
        s = np.array([1, 0, 0, 0], dtype=float)
        a, probs = ac.select_action(s, seed=0)
        self.assertIn(a, [0, 1])
        self.assertEqual(probs.shape, (2,))

    def test_update_returns_advantage(self):
        ac = main.ActorCritic(n_states=4, n_actions=2, hidden=8, seed=0)
        s = np.array([1, 0, 0, 0], dtype=float)
        a = 0
        s_next = np.array([0, 1, 0, 0], dtype=float)
        adv = ac.update(s, a, 1.0, s_next, done=False)
        self.assertIsInstance(adv, float)


class TestNStepAdvantage(unittest.TestCase):
    def test_simple(self):
        rewards = [1, 0, 0]
        values = [0, 0, 0, 0]
        adv = main.n_step_advantage(rewards, values, gamma=0.9, n=2, done_at=2)
        # n=2: A_0 = 1 + 0.9 * V(s_2) - V(s_0) = 1 + 0.9*0 - 0 = 1.0
        self.assertAlmostEqual(adv[0], 1.0, places=4)


class TestGAE(unittest.TestCase):
    def test_simple(self):
        # rewards=[1, 0], values=[0, 0, 0], dones=[0, 1], gamma=0.9, lambda=0.95
        rewards = [1, 0]
        values = [0, 0, 0]
        dones = [0, 1]
        adv = main.gae_advantage(rewards, values, dones, gamma=0.9, lambda_=0.95)
        # A_1 = 0 + 0.9*0*0.95*0 - 0 = 0
        # A_0 = (1 + 0.9*0 - 0) + 0.9*0.95*0*A_1 = 1.0
        self.assertAlmostEqual(adv[0], 1.0, places=4)
        self.assertAlmostEqual(adv[1], 0.0, places=4)


class TestSummary(unittest.TestCase):
    def test_cinco_keys(self):
        s = main.a2c_a3c_summary()
        self.assertEqual(len(s), 5)


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