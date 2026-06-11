"""Pruebas para 18-multi-token-prediction."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestMTPLoss(unittest.TestCase):
    def test_basic(self):
        main_logits = np.random.default_rng(0).standard_normal((4, 10))
        aux_logits = np.random.default_rng(1).standard_normal((4, 10))
        targets = np.array([0, 1, 2, 3])
        aux_targets = np.array([1, 2, 3, 4])
        loss = main.mtp_loss(main_logits, aux_logits, targets, aux_targets)
        self.assertGreater(loss, 0)


class TestSignalGain(unittest.TestCase):
    def test_k4(self):
        g = main.mtp_training_signal_gain(k_aux=4)
        self.assertEqual(g, 5)

    def test_k0(self):
        g = main.mtp_training_signal_gain(k_aux=0)
        self.assertEqual(g, 1)


class TestSpeedup(unittest.TestCase):
    def test_basic(self):
        sp = main.mtp_inference_speedup(aux_acceptance_rate=0.7, k_aux=4)
        self.assertGreater(sp, 1.0)

    def test_low_acceptance(self):
        sp = main.mtp_inference_speedup(aux_acceptance_rate=0.0, k_aux=4)
        # 0 acceptance, speedup = 1 / (k * 0.1 + 1)
        self.assertAlmostEqual(sp, 1.0 / 1.4, places=3)


class TestComponents(unittest.TestCase):
    def test_keys(self):
        c = main.mtp_components()
        self.assertIn("Main head", c)
        self.assertIn("Aux heads", c)


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