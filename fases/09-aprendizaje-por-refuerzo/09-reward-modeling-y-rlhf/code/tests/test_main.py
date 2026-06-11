"""Pruebas para 09-reward-modeling-y-rlhf."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestBradleyTerry(unittest.TestCase):
    def test_zero_si_chosen_mayor(self):
        loss = main.bradley_terry_loss(chosen_reward=10.0, rejected_reward=0.0)
        self.assertLess(loss, 0.01)

    def test_loss_positivo(self):
        loss = main.bradley_terry_loss(1.0, 0.0)
        # sigmoid(1) ~ 0.731, -log(0.731) ~ 0.31
        self.assertGreater(loss, 0.2)
        self.assertLess(loss, 0.4)

    def test_loss_mayor_si_chosen_rejected_iguales(self):
        loss1 = main.bradley_terry_loss(1.0, 0.0)
        loss2 = main.bradley_terry_loss(0.5, 0.5)
        # Si diff=0, sigmoid(0)=0.5, -log(0.5)=0.69 (mayor)
        self.assertGreater(loss2, loss1)


class TestPairwiseAccuracy(unittest.TestCase):
    def test_perfect(self):
        chosen = np.array([2, 1, 3])
        rejected = np.array([1, 0, 2])
        acc = main.pairwise_accuracy(chosen, rejected)
        self.assertEqual(acc, 1.0)

    def test_zero(self):
        chosen = np.array([0, 0, 0])
        rejected = np.array([1, 1, 1])
        acc = main.pairwise_accuracy(chosen, rejected)
        self.assertEqual(acc, 0.0)


class TestDPOLoss(unittest.TestCase):
    def test_zero_si_preferido(self):
        loss = main.dpo_loss(chosen_logp=0.0, rejected_logp=-2.0,
                              ref_chosen_logp=-1.0, ref_rejected_logp=-1.0)
        # chosen_diff=1, rejected_diff=-1, diff=2
        # beta * 2 = 0.2, sigmoid(0.2)=0.55, -log(0.55)=0.60
        self.assertLess(loss, 1.0)


class TestGRPO(unittest.TestCase):
    def test_shape(self):
        rewards = np.array([1.0, 0.5, 0.0, 2.0])
        loss = main.grpo_loss(rewards)
        self.assertIsInstance(loss, float)


class TestPipeline(unittest.TestCase):
    def test_cuatro_pasos(self):
        steps = main.rlhf_pipeline_steps()
        self.assertEqual(len(steps), 4)


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