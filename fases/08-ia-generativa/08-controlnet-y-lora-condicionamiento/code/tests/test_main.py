"""Pruebas para 08-controlnet-y-lora-condicionamiento."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestZeroConv(unittest.TestCase):
    def test_zero_init(self):
        W = np.random.default_rng(0).standard_normal((3, 4))
        b = np.zeros(4)
        W_z, b_z = main.zero_convolution(W, b, seed=0)
        # Init cercano a 0
        self.assertLessEqual(np.abs(W_z).max(), 0.01)
        self.assertEqual(b_z.shape, b.shape)


class TestLoRA(unittest.TestCase):
    def test_init_b_es_cero(self):
        A, B = main.lora_init(in_dim=8, out_dim=16, rank=4, seed=0)
        # B deberia ser 0
        np.testing.assert_array_equal(B, np.zeros_like(B))

    def test_init_a_no_es_cero(self):
        A, B = main.lora_init(in_dim=8, out_dim=16, rank=4, seed=0)
        self.assertGreater(np.abs(A).sum(), 0)

    def test_forward_init_es_identidad(self):
        # Inicialmente BA = 0, forward == x @ W
        W = np.random.default_rng(0).standard_normal((4, 8)) * 0.1
        A, B = main.lora_init(4, 8, rank=2, seed=0)
        x = np.random.default_rng(1).standard_normal((3, 4))
        out = main.lora_forward(x, W, A, B)
        expected = x @ W
        np.testing.assert_allclose(out, expected, atol=1e-9)

    def test_merge_equivalente(self):
        # forward == x @ (W + alpha/rank * A @ B)
        W = np.random.default_rng(0).standard_normal((4, 8)) * 0.1
        A, B = main.lora_init(4, 8, rank=2, seed=0)
        # Modificar B para que no sea 0
        B_actual = np.ones_like(B)
        x = np.random.default_rng(1).standard_normal((3, 4))
        # Forward con LoRA
        out_lora = main.lora_forward(x, W, A, B_actual, alpha=2.0)
        # Merge
        W_merged = main.merge_lora(W, A, B_actual, alpha=2.0)
        out_merged = x @ W_merged
        np.testing.assert_allclose(out_lora, out_merged, atol=1e-9)


class TestControlNetConditions(unittest.TestCase):
    def test_siete_conditions(self):
        conds = main.controlnet_conditions()
        self.assertEqual(len(conds), 7)


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