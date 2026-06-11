"""Pruebas para 14-construye-un-transformer-capstone."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestDecoderBlock(unittest.TestCase):
    def test_shape(self):
        block = main.DecoderBlock(d_model=8, n_heads=2, d_ff=16, seed=0)
        x = np.random.default_rng(0).standard_normal((3, 8))
        mask = main.causal_mask(3)
        out = block(x, mask)
        self.assertEqual(out.shape, (3, 8))


class TestTransformerLM(unittest.TestCase):
    def test_forward_shape(self):
        model = main.TransformerLM(vocab_size=50, d_model=16, n_heads=2,
                                   n_layers=2, d_ff=32, max_seq=20, seed=0)
        ids = np.array([1, 2, 3, 4])
        logits = model.forward(ids)
        self.assertEqual(logits.shape, (4, 50))

    def test_generate_extiende(self):
        model = main.TransformerLM(vocab_size=50, d_model=16, n_heads=2,
                                   n_layers=2, d_ff=32, max_seq=20, seed=42)
        prompt = [1, 2, 3]
        out = model.generate(prompt, max_new=5, temperature=1.0, seed=0)
        self.assertEqual(len(out), 8)
        # prompt deberia estar al inicio
        self.assertEqual(out[:3], [1, 2, 3])

    def test_generate_determinista(self):
        # Misma seed = misma salida
        model = main.TransformerLM(vocab_size=20, d_model=8, n_heads=2,
                                   n_layers=1, d_ff=16, max_seq=10, seed=0)
        out1 = model.generate([1, 2], max_new=3, temperature=0.1, seed=0)
        out2 = model.generate([1, 2], max_new=3, temperature=0.1, seed=0)
        self.assertEqual(out1, out2)

    def test_sinusaloidal_en_rango(self):
        model = main.TransformerLM(vocab_size=10, d_model=8, n_heads=2,
                                   n_layers=1, d_ff=16, max_seq=20, seed=0)
        # pos_emb deberia estar en [-1, 1]
        self.assertGreaterEqual(model.pos_emb.min(), -1.0001)
        self.assertLessEqual(model.pos_emb.max(), 1.0001)


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