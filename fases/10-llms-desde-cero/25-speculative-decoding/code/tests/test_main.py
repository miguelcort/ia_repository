"""Pruebas para 25-speculative-decoding."""
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
    def test_basic(self):
        p_target = np.array([0.5, 0.5])
        p_draft = np.array([0.5, 0.5])
        a = main.acceptance_rejection(p_target, p_draft, 0)
        self.assertEqual(a, 1.0)

    def test_parcial(self):
        p_target = np.array([0.4, 0.6])
        p_draft = np.array([0.5, 0.5])
        a = main.acceptance_rejection(p_target, p_draft, 0)
        # 0.4 / 0.5 = 0.8
        self.assertAlmostEqual(a, 0.8, places=3)

    def test_full_accept(self):
        p_target = np.array([0.9, 0.1])
        p_draft = np.array([0.1, 0.9])
        a = main.acceptance_rejection(p_target, p_draft, 0)
        # 0.9 / 0.1 = 9 -> min(1, 9) = 1
        self.assertEqual(a, 1.0)


class TestAdjusted(unittest.TestCase):
    def test_sum_to_one(self):
        p_target = np.array([0.5, 0.3, 0.2])
        p_draft = np.array([0.4, 0.4, 0.2])
        p_adj = main.adjusted_distribution(p_target, p_draft)
        np.testing.assert_allclose(p_adj.sum(), 1.0, atol=1e-6)
        # p_adj deberia ser >= 0
        self.assertTrue((p_adj >= 0).all())


class TestSpeedup(unittest.TestCase):
    def test_high_acceptance(self):
        sp = main.expected_speedup(0.9, k_draft=4)
        # High rate: deberia ser > 1.0
        self.assertGreater(sp, 1.5)

    def test_zero_acceptance(self):
        sp = main.expected_speedup(0.0, k_draft=4)
        # r=0: 0 accepted, speedup = 1 / 1.4
        self.assertAlmostEqual(sp, 1.0 / 1.4, places=3)

    def test_perfect_acceptance(self):
        sp = main.expected_speedup(0.999, k_draft=4)
        # Casi perfecto: ~5/1.4
        self.assertGreater(sp, 3.0)


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