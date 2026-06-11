"""Pruebas para 10-mecanismo-de-atencion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSDP(unittest.TestCase):
    def test_shape(self):
        T_q, T_k, d = 3, 4, 5
        rng = np.random.default_rng(0)
        Q = rng.normal(size=(T_q, d))
        K = rng.normal(size=(T_k, d))
        V = rng.normal(size=(T_k, d))
        out, pesos = main.scaled_dot_product_attention(Q, K, V)
        self.assertEqual(out.shape, (T_q, d))
        self.assertEqual(pesos.shape, (T_q, T_k))

    def test_pesos_suman_uno(self):
        T_q, T_k, d = 3, 4, 5
        rng = np.random.default_rng(0)
        Q = rng.normal(size=(T_q, d))
        K = rng.normal(size=(T_k, d))
        V = rng.normal(size=(T_k, d))
        _, pesos = main.scaled_dot_product_attention(Q, K, V)
        # Suma 1 por query
        sums = pesos.sum(axis=-1)
        for s in sums:
            self.assertAlmostEqual(float(s), 1.0, places=5)

    def test_mask_aplica(self):
        d = 4
        rng = np.random.default_rng(0)
        Q = rng.normal(size=(1, d))
        K = rng.normal(size=(3, d))
        V = rng.normal(size=(3, d))
        # Mascara: solo el primer key es visible
        mask = np.array([[1, 0, 0]])
        out, pesos = main.scaled_dot_product_attention(Q, K, V, mask=mask)
        # Atencion deberia estar concentrada en el primer key
        self.assertGreater(pesos[0, 0], 0.99)


class TestBahdanau(unittest.TestCase):
    def test_bahdanau(self):
        h_dec = np.array([1.0, 0.0])
        h_enc = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        ctx, pesos = main.bahdanau_attention(h_dec, h_enc)
        self.assertEqual(ctx.shape, (2,))
        # Peso mayor en el primer enc (mas similar)
        self.assertGreater(pesos[0], pesos[1])


class TestLuong(unittest.TestCase):
    def test_luong(self):
        h_dec = np.array([1.0, 0.0])
        h_enc = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        ctx, pesos = main.luong_attention(h_dec, h_enc)
        # Peso mayor en el primer enc
        self.assertGreater(pesos[0], pesos[1])


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("SDP", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()