"""Pruebas para 04-pre-training-mini-gpt."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCrossEntropy(unittest.TestCase):
    def test_perfect(self):
        logits = np.zeros((3, 5))
        targets = np.array([0, 1, 2])
        for i, t in enumerate(targets):
            logits[i, t] = 100
        loss = main.cross_entropy(logits, targets)
        self.assertLess(loss, 0.01)

    def test_uniforme(self):
        logits = np.zeros((3, 10))
        targets = np.array([0, 1, 2])
        loss = main.cross_entropy(logits, targets)
        # log(10) ~ 2.3
        self.assertGreater(loss, 2.0)
        self.assertLess(loss, 2.5)


class TestPerplexity(unittest.TestCase):
    def test_uniforme(self):
        logits = np.zeros((3, 10))
        targets = np.array([0, 1, 2])
        ppl = main.perplexity(logits, targets)
        np.testing.assert_allclose(ppl, 10.0, rtol=0.1)


class TestLRSchedule(unittest.TestCase):
    def test_warmup(self):
        lr = main.lr_schedule(0, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        # step 0, warmup lr = max_lr * 1/100
        self.assertAlmostEqual(lr, 1e-3 * 1 / 100, places=7)

    def test_cosine_end(self):
        lr = main.lr_schedule(1000, warmup_steps=100, max_steps=1000, max_lr=1e-3)
        # At max_steps, lr = min_lr
        self.assertAlmostEqual(lr, 1e-4, places=7)  # min_lr default


class TestAdamW(unittest.TestCase):
    def test_update_shape(self):
        p = np.ones((3, 3))
        g = np.full((3, 3), 0.5)
        m = np.zeros_like(p)
        v = np.zeros_like(p)
        p_new, m_new, v_new = main.adamw_step(p, g, m, v, t=1, weight_decay=0.0, lr=1e-1)
        self.assertEqual(p_new.shape, p.shape)
        # m, v deberian haber cambiado
        self.assertNotEqual(m_new[0, 0], 0)
        self.assertNotEqual(v_new[0, 0], 0)
        # p_new deberia tener forma
        self.assertFalse(np.any(np.isnan(p_new)))


class TestGradClip(unittest.TestCase):
    def test_clip(self):
        g = np.full(10, 100.0)  # norm 1000
        clipped = main.grad_clip(g, max_norm=1.0)
        np.testing.assert_allclose(np.linalg.norm(clipped), 1.0, rtol=1e-6)


class TestConfig(unittest.TestCase):
    def test_config(self):
        c = main.pretraining_config()
        self.assertEqual(c["d_model"], 768)
        self.assertIn("n_heads", c)


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