"""Pruebas para 04-flamingo-gated-cross-attention."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestGatedCrossAttn(unittest.TestCase):
    def test_zero_gate_identity(self):
        # gate=0 -> tanh(0)=0 -> out = h + 0 = h
        rng = np.random.default_rng(0)
        h = rng.standard_normal((4, 8)) * 0.1
        kv = rng.standard_normal((10, 8)) * 0.1
        Wq = rng.standard_normal((8, 8)) * 0.1
        Wkv = rng.standard_normal((8, 8)) * 0.1
        Wo = rng.standard_normal((8, 8)) * 0.1
        out = main.gated_cross_attention(h, kv, Wq, Wkv, Wo, gate_logit_zero_init=True)
        np.testing.assert_allclose(out, h, atol=1e-10)

    def test_mismatched_dim(self):
        # text 16 dim, vision 8 dim
        rng = np.random.default_rng(0)
        h = rng.standard_normal((4, 16)) * 0.1
        kv = rng.standard_normal((10, 8)) * 0.1
        Wq = rng.standard_normal((16, 16)) * 0.1
        Wkv = rng.standard_normal((16, 16)) * 0.1
        Wo = rng.standard_normal((16, 16)) * 0.1
        out = main.gated_cross_attention(h, kv, Wq, Wkv, Wo, gate_logit_zero_init=False)
        self.assertEqual(out.shape, (4, 16))


class TestFlamingoBlock(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        h = rng.standard_normal((5, 32)) * 0.1
        kv = rng.standard_normal((20, 32)) * 0.1
        Wq = rng.standard_normal((32, 32)) * 0.1
        Wkv = rng.standard_normal((32, 32)) * 0.1
        Wo = rng.standard_normal((32, 32)) * 0.1
        Wf1 = rng.standard_normal((32, 128)) * 0.1
        Wf2 = rng.standard_normal((128, 32)) * 0.1
        out = main.flamingo_block(h, kv, Wq, Wkv, Wo, Wf1, Wf2)
        self.assertEqual(out.shape, (5, 32))


class TestFlamingoForward(unittest.TestCase):
    def test_alternates_vision(self):
        rng = np.random.default_rng(0)
        text = rng.standard_normal((4, 16)) * 0.1
        v1 = rng.standard_normal((8, 16)) * 0.1
        v2 = rng.standard_normal((6, 16)) * 0.1
        out = main.flamingo_forward(text, [v1, v2], n_layers=4, d=16)
        self.assertEqual(out.shape, (4, 16))


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