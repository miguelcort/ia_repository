"""Pruebas para 09-secuencia-a-secuencia."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEncoder(unittest.TestCase):
    def test_shape(self):
        H, D = 8, 4
        rng = np.random.default_rng(0)
        W_hh = rng.normal(scale=0.1, size=(H, H))
        W_xh = rng.normal(scale=0.1, size=(H, D))
        b = np.zeros(H)
        h0 = np.zeros(H)
        x = [rng.normal(size=D) for _ in range(5)]
        h = main.encoder_rnn(x, h0, W_hh, W_xh, b)
        self.assertEqual(h.shape, (H,))


class TestDecoder(unittest.TestCase):
    def test_step(self):
        V, H, D = 10, 4, 8
        rng = np.random.default_rng(0)
        W_hh = rng.normal(scale=0.1, size=(H, H))
        W_xh = rng.normal(scale=0.1, size=(H, D))
        W_hy = rng.normal(scale=0.1, size=(V, H))
        b_h = np.zeros(H)
        b_y = np.zeros(V)
        y_prev = np.zeros(V)
        h_prev = np.zeros(H)
        h, probs, idx, tok = main.decoder_rnn_step(y_prev, h_prev, W_hh, W_xh, W_hy, b_h, b_y, ["<sos>", "a", "b", "<eos>"])
        self.assertEqual(probs.shape, (V,))
        # Suma 1
        self.assertAlmostEqual(float(probs.sum()), 1.0)


class TestSeq2Seq(unittest.TestCase):
    def test_pipeline(self):
        V, H, D = 10, 4, 8
        rng = np.random.default_rng(0)
        W_hh = rng.normal(scale=0.1, size=(H, H))
        W_xh = rng.normal(scale=0.1, size=(H, D))
        W_hy = rng.normal(scale=0.1, size=(V, H))
        b_h = np.zeros(H)
        b_y = np.zeros(V)
        h0 = np.zeros(H)
        x = [rng.normal(size=D) for _ in range(3)]
        out = main.seq2seq(x, h0, W_hh, W_xh, W_hy, b_h, b_y, ["<sos>", "a", "b", "<eos>"], max_len=3)
        self.assertGreater(len(out), 0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Output", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()