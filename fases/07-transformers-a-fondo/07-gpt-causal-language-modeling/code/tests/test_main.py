"""Pruebas para 07-gpt-causal-language-modeling."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCausalMask(unittest.TestCase):
    def test_forma(self):
        m = main.causal_mask(4)
        self.assertEqual(m.shape, (4, 4))

    def test_bloquea_futuro(self):
        m = main.causal_mask(3)
        # Triangular superior = -inf (futuro)
        self.assertEqual(m[0, 1], -1e9)
        self.assertEqual(m[0, 2], -1e9)
        # Pasado y self: 0
        self.assertEqual(m[0, 0], 0)
        self.assertEqual(m[1, 0], 0)
        self.assertEqual(m[1, 1], 0)


class TestLoss(unittest.TestCase):
    def test_perfecta(self):
        # logits.argmax = target -> loss ~ 0
        logits = np.zeros((3, 5))
        targets = np.array([0, 1, 2])
        for i, t in enumerate(targets):
            logits[i, t] = 100
        loss = main.cross_entropy_loss(logits, targets)
        self.assertLess(loss, 0.01)

    def test_perplexidad_uniforme(self):
        # logits uniformes -> PPL = vocab_size
        logits = np.zeros((5, 10))
        targets = np.array([0, 1, 2, 3, 4])
        ppl = main.perplexity(logits, targets)
        np.testing.assert_allclose(ppl, 10.0, rtol=1e-3)


class TestTopK(unittest.TestCase):
    def test_conserva_top_k(self):
        logits = np.array([1, 5, 2, 8, 3, 7, 4])
        filtered = main.top_k_filter(logits, k=3)
        # Top 3 son 8, 7, 5
        self.assertEqual(filtered[3], 8)
        self.assertEqual(filtered[5], 7)
        self.assertEqual(filtered[1], 5)
        # Otros son -inf
        for i in [0, 2, 4, 6]:
            self.assertEqual(filtered[i], -1e9)


class TestTopP(unittest.TestCase):
    def test_nucleus_chico(self):
        # Si un token domina, mantener solo ese
        logits = np.array([1.0, 100.0, 1.0, 1.0, 1.0])
        filtered = main.top_p_filter(logits, p=0.5)
        # Solo el token dominante deberia mantenerse
        self.assertGreater(filtered[1], -1e8)
        self.assertEqual(filtered[0], -1e9)


class TestSampling(unittest.TestCase):
    def test_greedy_temperatura_baja(self):
        # Con T=0.01, deberia ser determinista
        rng = np.random.default_rng(0)
        logits = np.array([1.0, 5.0, 2.0, 3.0])
        for _ in range(3):
            tok = main.sample_next_token(logits, temperature=0.01, seed=0)
            self.assertEqual(tok, 1)  # el mayor

    def test_sampling_rango_valido(self):
        logits = np.random.default_rng(0).standard_normal(10)
        for _ in range(10):
            tok = main.sample_next_token(logits, temperature=1.0, k=5, p=0.9, seed=0)
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, 10)


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