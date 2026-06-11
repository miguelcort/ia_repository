"""Pruebas para 02-self-attention-desde-cero."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestSoftmax(unittest.TestCase):
    def test_suma_uno(self):
        x = np.array([[1.0, 2.0, 3.0]])
        out = main.softmax(x)
        self.assertAlmostEqual(out.sum(), 1.0, places=6)

    def test_estabilidad_numerica(self):
        # Sin restar max, exp(1000) = inf
        x = np.array([[1000.0, 1001.0, 1002.0]])
        out = main.softmax(x)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertAlmostEqual(out.sum(), 1.0, places=6)


class TestSelfAttention(unittest.TestCase):
    def test_output_shape(self):
        rng = np.random.default_rng(0)
        Q, K, V = [rng.standard_normal((5, 8)) for _ in range(3)]
        out, w = main.self_attention(Q, K, V)
        self.assertEqual(out.shape, (5, 8))
        self.assertEqual(w.shape, (5, 5))

    def test_attention_suma_uno_por_fila(self):
        rng = np.random.default_rng(1)
        Q, K, V = [rng.standard_normal((4, 6)) for _ in range(3)]
        _, w = main.self_attention(Q, K, V)
        # Cada fila de attention weights suma ~1
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-6)

    def test_scaling_importa(self):
        # Si d_k crece, scores crecen, softmax mas picuda (concentrada)
        rng = np.random.default_rng(2)
        Q, K = rng.standard_normal((10, 256)), rng.standard_normal((10, 256))
        V = np.zeros((10, 256))
        _, w_scaling = main.self_attention(Q, K, V)
        # Una sola fila deberia dominar (concentracion por escala)
        max_w = w_scaling.max(axis=-1)
        self.assertGreater(max_w.max(), 0.5)

    def test_causal_mask_bloquea_futuro(self):
        # Token 0 solo debe atender a si mismo (peso ~= 1)
        seq = 5
        rng = np.random.default_rng(3)
        Q, K, V = [rng.standard_normal((seq, 4)) for _ in range(3)]
        _, w = main.self_attention(Q, K, V, mask=main.causal_mask(seq))
        # w[0, 1:] deberian ser ~0
        np.testing.assert_allclose(w[0, 1:], 0.0, atol=1e-9)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Pesos", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()