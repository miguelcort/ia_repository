"""Pruebas para 15-speculative-decoding-eagle3."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestAcceptance(unittest.TestCase):
    def test_accept_si_target_mayor(self):
        r = main.speculative_acceptance(target_probs=np.array([0.5, 0.5]),
                                         draft_probs=np.array([0.5, 0.5]),
                                         draft_token=0)
        self.assertEqual(r, 1.0)

    def test_parcial_si_target_menor(self):
        r = main.speculative_acceptance(target_probs=np.array([0.4, 0.6]),
                                         draft_probs=np.array([0.5, 0.5]),
                                         draft_token=0)
        # 0.4 / 0.5 = 0.8
        self.assertAlmostEqual(r, 0.8, places=3)


class TestSpeedup(unittest.TestCase):
    def test_perfect(self):
        # Acceptance 1.0: acepta todos
        sp = main.eagle3_speedup(1.0, k_draft_tokens=5)
        self.assertGreater(sp, 1.0)

    def test_zero_acceptance(self):
        sp = main.eagle3_speedup(0.0, k_draft_tokens=5)
        # Sin acceptance, speedup = 1 / (k * 0.1 + 1) = 1/1.5
        self.assertAlmostEqual(sp, 1.0 / 1.5, places=3)


class TestComponents(unittest.TestCase):
    def test_seis(self):
        c = main.eagle3_components()
        self.assertEqual(len(c), 6)


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