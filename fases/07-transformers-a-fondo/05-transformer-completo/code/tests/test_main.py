"""Pruebas para 05-transformer-completo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestLayerNorm(unittest.TestCase):
    def test_normaliza(self):
        # Output tiene media 0, var 1 (por fila)
        x = np.random.default_rng(0).standard_normal((3, 10))
        out = main.layer_norm(x)
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-3)


class TestFFN(unittest.TestCase):
    def test_shape(self):
        x = np.random.default_rng(1).standard_normal((4, 8))
        W1 = np.random.default_rng(2).standard_normal((8, 16)) * 0.1
        b1 = np.zeros(16)
        W2 = np.random.default_rng(3).standard_normal((16, 8)) * 0.1
        b2 = np.zeros(8)
        out = main.ffn(x, W1, b1, W2, b2)
        self.assertEqual(out.shape, (4, 8))


class TestEncoderBlock(unittest.TestCase):
    def test_shape(self):
        seq, d, d_ff = 4, 8, 32
        x = np.random.default_rng(4).standard_normal((seq, d))
        weights = main.init_block(d, d_ff, seed=4)
        out = main.encoder_block(x, *weights)
        self.assertEqual(out.shape, (seq, d))

    def test_residual_connection(self):
        # Cambiar pesos attention no deberia causar gradient explosion
        seq, d, d_ff = 3, 4, 8
        x = np.random.default_rng(5).standard_normal((seq, d))
        weights1 = main.init_block(d, d_ff, seed=5)
        weights2 = main.init_block(d, d_ff, seed=999)
        out1 = main.encoder_block(x, *weights1)
        out2 = main.encoder_block(x, *weights2)
        # Output magnitudes razonables
        self.assertLess(np.abs(out1).max(), 100)
        self.assertLess(np.abs(out2).max(), 100)


class TestDecoderBlock(unittest.TestCase):
    def test_shape(self):
        seq, d, d_ff = 4, 8, 32
        x = np.random.default_rng(6).standard_normal((seq, d))
        enc = np.random.default_rng(7).standard_normal((seq, d))
        cm = np.triu(np.ones((seq, seq)) * -1e9, k=1)
        # Necesita pesos para self-attn Y cross-attn Y FFN
        W_Q, W_K, W_V, W_O, W1, b1, W2, b2 = main.init_block(d, d_ff, seed=6)
        W_Q2, W_K2, W_V2, W_O2 = main.init_block(d, d_ff, seed=7)[:4]
        out = main.decoder_block(x, enc, W_Q, W_K, W_V, W_O,
                                 W_Q2, W_K2, W_V2, W_O2,
                                 W1, b1, W2, b2, cm)
        self.assertEqual(out.shape, (seq, d))


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