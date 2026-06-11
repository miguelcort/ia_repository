"""Pruebas para 08-cnns-y-rnns-para-texto."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestTextCNN(unittest.TestCase):
    def test_shape(self):
        x = np.random.default_rng(0).normal(size=(10, 8))
        kernels = [np.random.default_rng(i).normal(size=(3, 8)) for i in range(3)]
        feats = main.text_cnn_kernel(x, kernels)
        # 1 feature por kernel
        self.assertEqual(feats.shape, (3,))

    def test_kernels_diferentes(self):
        x = np.random.default_rng(0).normal(size=(10, 8))
        k1 = np.zeros((3, 8))
        k2 = np.ones((3, 8))
        feats = main.text_cnn_kernel(x, [k1, k2])
        # k1 da 0, k2 da max pool de 3*8=24
        self.assertNotEqual(feats[0], feats[1])


class TestRNNCell(unittest.TestCase):
    def test_shape(self):
        x_t = np.random.default_rng(0).normal(size=8)
        h_prev = np.zeros(16)
        W_hh = np.random.default_rng(0).normal(scale=0.1, size=(16, 16))
        W_xh = np.random.default_rng(1).normal(scale=0.1, size=(16, 8))
        b = np.zeros(16)
        h = main.rnn_cell_simple(x_t, h_prev, W_hh, W_xh, b)
        self.assertEqual(h.shape, (16,))

    def test_tanh_rango(self):
        x_t = np.zeros(4)
        h_prev = np.zeros(4)
        W_hh = np.eye(4) * 0.5
        W_xh = np.zeros((4, 4))
        b = np.zeros(4)
        h = main.rnn_cell_simple(x_t, h_prev, W_hh, W_xh, b)
        # tanh(0) = 0
        np.testing.assert_array_almost_equal(h, np.zeros(4))


class TestBiRNN(unittest.TestCase):
    def test_shape(self):
        T, D, H = 5, 4, 8
        x = np.random.default_rng(0).normal(size=(T, D))
        h0 = np.zeros(H)
        bi = main.bidirectional_rnn(x, h0, np.eye(H) * 0.1, np.zeros((H, D)),
                                    np.eye(H) * 0.1, np.zeros((H, D)),
                                    np.zeros(H), np.zeros(H))
        # Concat forward + backward -> 2H
        self.assertEqual(bi.shape, (T, 2 * H))


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("TextCNN", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()