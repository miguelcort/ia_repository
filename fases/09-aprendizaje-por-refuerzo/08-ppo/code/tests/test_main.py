"""Pruebas para 08-ppo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPPOClipped(unittest.TestCase):
    def test_no_clip_si_ratio_cerca_uno(self):
        # ratio=1, advantage > 0 -> L = 1 * A
        ratio = np.array([1.0])
        advantage = np.array([2.0])
        out = main.ppo_clipped_objective(ratio, advantage, clip=0.2)
        np.testing.assert_array_equal(out, [2.0])

    def test_clip_si_ratio_mayor_1eps(self):
        # ratio=1.5 > 1+0.2, advantage > 0
        ratio = np.array([1.5])
        advantage = np.array([2.0])
        out = main.ppo_clipped_objective(ratio, advantage, clip=0.2)
        # Clipped a 1.2, L = 1.2 * 2 = 2.4
        self.assertAlmostEqual(out[0], 2.4, places=4)

    def test_clip_si_ratio_menor_y_advantage_negativo(self):
        # ratio=0.5 < 1-0.2, advantage < 0
        ratio = np.array([0.5])
        advantage = np.array([-2.0])
        out = main.ppo_clipped_objective(ratio, advantage, clip=0.2)
        # r*A = -1.0, clipped = 0.8 * -2 = -1.6, min = -1.6
        self.assertAlmostEqual(out[0], -1.6, places=4)


class TestKLPenalty(unittest.TestCase):
    def test_zero_si_identica(self):
        pi = np.array([0.5, 0.3, 0.2])
        kl = main.kl_penalty(pi, pi)
        self.assertAlmostEqual(kl, 0.0, places=6)


class TestPPOLoss(unittest.TestCase):
    def test_loss_shape(self):
        advantages = np.array([1.0, -0.5])
        old_log_probs = np.zeros(2)
        new_log_probs = np.array([0.1, -0.1])
        loss = main.ppo_clip_loss(advantages, old_log_probs, new_log_probs)
        self.assertIsInstance(loss, float)


class TestGAE(unittest.TestCase):
    def test_simple(self):
        rewards = [1, 0]
        values = [0, 0, 0]
        dones = [0, 1]
        adv = main.compute_advantages(rewards, values, dones, gamma=0.9, lambda_=0.95)
        self.assertEqual(len(adv), 2)


class TestComponents(unittest.TestCase):
    def test_cinco_components(self):
        c = main.ppo_components()
        self.assertEqual(len(c), 5)

    def test_algorithm_cinco_steps(self):
        a = main.ppo_algorithm()
        self.assertEqual(len(a), 5)


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