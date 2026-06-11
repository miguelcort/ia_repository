"""Pruebas para 16-speculative-decoding."""
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
        # Si target prob > draft prob, accept con prob 1
        p_draft = np.array([0.5, 0.5])
        p_target = np.array([0.8, 0.2])
        # Token 0: target/draft = 0.8/0.5 = 1.6 -> min(1, 1.6) = 1
        r = main.acceptance_rejection(p_draft, p_target, 0)
        self.assertEqual(r, 1.0)

    def test_accept_parcial_si_draft_mayor(self):
        # Si draft > target, accept con prob < 1
        p_draft = np.array([0.8, 0.2])
        p_target = np.array([0.5, 0.5])
        # Token 0: target/draft = 0.5/0.8 = 0.625
        r = main.acceptance_rejection(p_draft, p_target, 0)
        self.assertAlmostEqual(r, 0.625, places=3)


class TestAdjustDistribution(unittest.TestCase):
    def test_suma_uno(self):
        p_draft = np.array([0.4, 0.3, 0.3])
        p_target = np.array([0.5, 0.2, 0.3])
        p_adj = main.adjust_distribution(p_draft, p_target, 0, gamma=4)
        np.testing.assert_allclose(p_adj.sum(), 1.0, atol=1e-6)

    def test_no_negativos(self):
        p_draft = np.array([0.4, 0.3, 0.3])
        p_target = np.array([0.5, 0.2, 0.3])
        p_adj = main.adjust_distribution(p_draft, p_target, 0, gamma=4)
        self.assertTrue((p_adj >= 0).all())


class TestSpeculativeDecoder(unittest.TestCase):
    def test_step_genera_tokens(self):
        decoder = main.SpeculativeDecoder(main.mock_draft_sampler,
                                           main.mock_target_sampler, gamma=4)
        out = decoder.step([1, 2, 3], seed=0)
        # Acepta o rechaza, en cualquier caso genera >= 1 token
        self.assertGreaterEqual(len(out), 1)
        self.assertLessEqual(len(out), 5)  # gamma + 1 max

    def test_maximo_gamma_mas_uno(self):
        decoder = main.SpeculativeDecoder(main.mock_draft_sampler,
                                           main.mock_target_sampler, gamma=3)
        out = decoder.step([1, 2, 3], seed=0)
        self.assertLessEqual(len(out), 4)


class TestBenchmark(unittest.TestCase):
    def test_speedup_mayor_que_uno(self):
        # Con draft 10x mas rapido y gamma=4, speedup > 1
        speedup = main.benchmark_speedup(seq_len=100, gamma=4)
        self.assertGreater(speedup, 1.0)


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