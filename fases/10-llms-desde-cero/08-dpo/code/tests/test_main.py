"""Pruebas para 08-dpo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDPOLoss(unittest.TestCase):
    def test_basic(self):
        loss = main.dpo_loss(0.0, -2.0, -1.0, -1.0)
        # chosen_diff=1, rejected_diff=-1, margin = 0.1 * 2 = 0.2
        # sigmoid(0.2) = 0.55, -log(0.55) = 0.60
        self.assertGreater(loss, 0)
        self.assertLess(loss, 1.0)

    def test_low_loss_when_preferred(self):
        # Strong margin
        loss = main.dpo_loss(2.0, -2.0, 0.0, 0.0)
        # margin = 0.1 * 4 = 0.4, -log(sigmoid(0.4)) = ~0.51
        self.assertLess(loss, 0.6)

    def test_vectorized(self):
        chosen = np.array([0.0, -1.0])
        rejected = np.array([-2.0, -3.0])
        ref_chosen = np.array([-1.0, -1.0])
        ref_rejected = np.array([-1.0, -1.0])
        loss = main.dpo_loss_vectorized(chosen, rejected, ref_chosen, ref_rejected)
        self.assertGreater(loss, 0)


class TestIPO(unittest.TestCase):
    def test_basic(self):
        loss = main.ipo_loss(1.0, -1.0, 0.0, 0.0)
        # diff - 1/(2*0.1) = 2 - 5 = -3, squared = 9
        self.assertGreater(loss, 0)


class TestKTO(unittest.TestCase):
    def test_basic(self):
        loss = main.kto_loss(0.0, -2.0, 0.0, 0.0)
        # chosen_diff=0, rejected_diff=-2
        # kl_rejected - kl_chosen = sigmoid(-2) - sigmoid(0) = 0.12 - 0.5 = -0.38
        self.assertIsInstance(loss, float)


class TestComponents(unittest.TestCase):
    def test_components(self):
        c = main.dpo_components()
        self.assertEqual(len(c), 5)

    def test_vs_ppo(self):
        c = main.dpo_vs_ppo()
        self.assertIn("DPO", c)
        self.assertIn("PPO", c)


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