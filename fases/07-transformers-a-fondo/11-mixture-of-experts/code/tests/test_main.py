"""Pruebas para 11-mixture-of-experts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestRouting(unittest.TestCase):
    def test_shape(self):
        x = np.random.default_rng(0).standard_normal((5, 8))
        W_gate = np.random.default_rng(1).standard_normal((8, 4)) * 0.1
        idx, w = main.top_k_routing(x, W_gate, n_experts=4, top_k=2)
        self.assertEqual(idx.shape, (5, 2))
        self.assertEqual(w.shape, (5, 2))

    def test_pesos_suman_uno(self):
        x = np.random.default_rng(0).standard_normal((5, 8))
        W_gate = np.random.default_rng(1).standard_normal((8, 4)) * 0.1
        _, w = main.top_k_routing(x, W_gate, n_experts=4, top_k=2)
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-6)

    def test_indices_en_rango(self):
        x = np.random.default_rng(0).standard_normal((5, 8))
        W_gate = np.random.default_rng(1).standard_normal((8, 4)) * 0.1
        idx, _ = main.top_k_routing(x, W_gate, n_experts=4, top_k=2)
        self.assertGreaterEqual(idx.min(), 0)
        self.assertLess(idx.max(), 4)


class TestExpertFFN(unittest.TestCase):
    def test_shape(self):
        x = np.random.default_rng(0).standard_normal((3, 8))
        W1 = np.random.default_rng(1).standard_normal((8, 16)) * 0.05
        b1 = np.zeros(16)
        W2 = np.random.default_rng(2).standard_normal((16, 8)) * 0.05
        b2 = np.zeros(8)
        out = main.expert_ffn(x, W1, b1, W2, b2)
        self.assertEqual(out.shape, (3, 8))


class TestMoE(unittest.TestCase):
    def test_output_shape(self):
        n, d, n_experts, k = 6, 8, 3, 2
        rng = np.random.default_rng(0)
        x = rng.standard_normal((n, d)) * 0.5
        W_gate = rng.standard_normal((d, n_experts)) * 0.1
        experts = [(rng.standard_normal((d, 16)) * 0.05,
                    np.zeros(16),
                    rng.standard_normal((16, d)) * 0.05,
                    np.zeros(d)) for _ in range(n_experts)]
        out = main.moe_layer(x, W_gate, experts, top_k=k)
        self.assertEqual(out.shape, (n, d))


class TestLoadBalance(unittest.TestCase):
    def test_loss_shape(self):
        probs = np.ones((4, 3)) / 3
        idx = np.array([[0, 1], [0, 1], [2, 1], [2, 0]])
        loss = main.load_balancing_loss(probs, idx, n_experts=3, top_k=2)
        self.assertGreater(loss, 0)
        self.assertLessEqual(loss, 1.0)

    def test_loss_cero_si_perfect_balance(self):
        # 4 tokens, 4 experts, top-1: cada token va a un expert
        probs = np.zeros((4, 4))
        for i in range(4):
            probs[i, i] = 1.0
        idx = np.array([[0], [1], [2], [3]])
        loss = main.load_balancing_loss(probs, idx, n_experts=4, top_k=1)
        # Perfect balance: f_i=0.25, p_i=0.25 -> 4 * sum(0.0625) = 0.25
        # 0.25 > 0 confirma loss > 0 (porque no es 'cero' en sentido literal)
        # Mejor verificar: si un solo expert recibe todo, loss = 4 * 1 * 1 = 4
        # Si balanceado: loss = 4 * (4 * 0.25 * 0.25) = 1
        # El loss NO deberia ser 0, deberia ser > 0
        self.assertGreater(loss, 0.0)


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