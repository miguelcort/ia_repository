"""Pruebas para 05-dqn."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDQN(unittest.TestCase):
    def test_init(self):
        agent = main.DQN(n_states=4, n_actions=2, hidden=8, seed=0)
        self.assertEqual(agent.n_states, 4)
        self.assertEqual(agent.n_actions, 2)
        self.assertEqual(agent.epsilon, 1.0)

    def test_forward_shape(self):
        agent = main.DQN(n_states=4, n_actions=2, hidden=8, seed=0)
        x = np.array([1.0, 0, 0, 0])
        q = agent.forward(x)
        self.assertEqual(q.shape, (2,))

    def test_act_argmax(self):
        agent = main.DQN(n_states=4, n_actions=2, hidden=8, seed=0)
        agent.epsilon = 0.0
        # Forzar argmax via W2 (shape hidden, n_actions) = (8, 2)
        # columna 1 grande, columna 0 chica
        new_W2 = np.zeros((8, 2), dtype=float)
        new_W2[:, 1] = 100.0
        agent.W2 = new_W2
        x = np.array([1.0, 0, 0, 0])
        a = agent.act(x)
        self.assertEqual(a, 1)

    def test_replay_buffer(self):
        agent = main.DQN(n_states=4, n_actions=2, hidden=8, seed=0)
        for i in range(5):
            agent.store((np.zeros(4), 0, 0.0, np.zeros(4), False))
        self.assertEqual(len(agent.replay), 5)

    def test_target_update(self):
        agent = main.DQN(n_states=4, n_actions=2, hidden=8, seed=0)
        # Modificar W1
        agent.W1[0, 0] = 99.0
        agent.update_target()
        # W1_t deberia copiar W1
        self.assertEqual(agent.W1_t[0, 0], 99.0)

    def test_decay_epsilon(self):
        agent = main.DQN(n_states=4, n_actions=2, hidden=8, seed=0)
        agent.epsilon = 1.0
        agent.decay_epsilon(decay=0.5)
        self.assertEqual(agent.epsilon, 0.5)
        agent.decay_epsilon(decay=0.5, min_eps=0.3)
        # 0.5 * 0.5 = 0.25 < 0.3, so 0.3
        self.assertEqual(agent.epsilon, 0.3)


class TestPrioritized(unittest.TestCase):
    def test_sum_to_one(self):
        td = np.array([1.0, 2.0, 3.0])
        p = main.prioritized_experience_weights(td)
        np.testing.assert_allclose(p.sum(), 1.0, atol=1e-6)
        # Mas alto TD error -> mas probability
        self.assertGreater(p[2], p[0])


class TestComponents(unittest.TestCase):
    def test_seis_componentes(self):
        comps = main.dqn_components()
        self.assertEqual(len(comps), 6)


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