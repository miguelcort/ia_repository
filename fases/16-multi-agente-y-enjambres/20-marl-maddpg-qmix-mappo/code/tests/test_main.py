"""Pruebas para 20-marl-maddpg-qmix-mappo."""
from __future__ import annotations
import random
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMultiAgentEnv(unittest.TestCase):
    def test_reset(self):
        random.seed(42)
        env = main.MultiAgentEnv(n_agents=2, n_actions=4)
        s = env.reset()
        self.assertEqual(len(s), 2)

    def test_step(self):
        env = main.MultiAgentEnv(n_agents=2, n_actions=4)
        env.reset()
        actions = [0, 1]
        next_s, rewards, done = env.step(actions)
        self.assertEqual(len(rewards), 2)


class TestMADDPGAgent(unittest.TestCase):
    def test_select_action(self):
        a = main.MADDPGAgent(0, n_actions=4)
        action = a.select_action(0.5, epsilon=0.0)
        self.assertIn(action, [0, 1, 2, 3])

    def test_select_action_random(self):
        a = main.MADDPGAgent(0, n_actions=4)
        action = a.select_action(0.5, epsilon=1.0)
        self.assertIn(action, [0, 1, 2, 3])

    def test_update(self):
        a = main.MADDPGAgent(0, n_actions=4)
        a.update(0.5, 0, 1.0, 0.5)
        self.assertNotEqual(a.q_values[0], 0.0)


class TestQMIX(unittest.TestCase):
    def test_mix(self):
        m = main.QMIXMixer(n_agents=3)
        result = m.mix([1.0, 2.0, 3.0])
        self.assertAlmostEqual(result, 2.0, places=4)


class TestTrain(unittest.TestCase):
    def test_maddpg(self):
        random.seed(42)
        env = main.MultiAgentEnv(n_agents=2, n_actions=4)
        rewards = main.maddpg_train(env, n_episodes=5, max_steps=3)
        self.assertEqual(len(rewards), 5)

    def test_mappo(self):
        random.seed(42)
        env = main.MultiAgentEnv(n_agents=2, n_actions=4)
        history = main.mappo_train(env, n_episodes=5, max_steps=3)
        self.assertEqual(len(history), 5)


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