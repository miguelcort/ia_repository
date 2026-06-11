"""Pruebas para 01-mdps-estados-acciones-y-recompensas."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDiscountedReturn(unittest.TestCase):
    def test_no_discount(self):
        # gamma=1, suma directa
        g = main.discounted_return([1, 2, 3], gamma=1.0)
        self.assertEqual(g, 6.0)

    def test_with_discount(self):
        # gamma=0.5: 1 + 0.5*2 + 0.25*3 = 1 + 1 + 0.75 = 2.75
        g = main.discounted_return([1, 2, 3], gamma=0.5)
        self.assertAlmostEqual(g, 2.75, places=4)

    def test_zero_rewards(self):
        g = main.discounted_return([0, 0, 0], gamma=0.9)
        self.assertEqual(g, 0.0)


class TestPolicyEvaluation(unittest.TestCase):
    def test_simple_mdp(self):
        n_s, n_a = 3, 2
        P = np.zeros((n_s, n_a, n_s))
        P[0, 0, 0] = 0.5; P[0, 0, 1] = 0.5
        P[0, 1, 1] = 1.0
        P[1, 0, 0] = 0.5; P[1, 0, 2] = 0.5
        P[1, 1, 2] = 1.0
        P[2, :, 2] = 1.0
        # Reward al transicionar a state 2 (cualquier accion en state 1)
        R = np.array([[0, 0], [1, 1], [0, 0]])
        policy = np.array([0, 0, 0])  # policy lleva a state 2 desde state 1
        V = main.policy_evaluation(P, R, policy, gamma=0.9, n_states=n_s, n_actions=n_a)
        # V[2] deberia ser ~0 (state absorbing sin reward adicional)
        self.assertLess(V[2], 0.1)
        # V[1] deberia ser > 0 (recolecta reward al transicionar)
        self.assertGreater(V[1], 0)


class TestPolicyImprovement(unittest.TestCase):
    def test_improvement(self):
        n_s, n_a = 3, 2
        P = np.zeros((n_s, n_a, n_s))
        P[0, 0, 0] = 0.5; P[0, 0, 1] = 0.5
        P[0, 1, 1] = 1.0
        P[1, 0, 2] = 0.5; P[1, 0, 1] = 0.5
        P[1, 1, 2] = 1.0
        P[2, :, 2] = 1.0
        R = np.array([[0, 0], [0, 0], [1, 1]])
        V = np.array([0.5, 0.8, 1.0])
        policy = main.policy_improvement(P, R, V, gamma=0.9, n_states=n_s, n_actions=n_a)
        # Verifica que policy es valida
        for s in range(n_s):
            self.assertIn(policy[s], [0, 1])


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