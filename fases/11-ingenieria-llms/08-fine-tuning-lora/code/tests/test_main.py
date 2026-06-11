"""Pruebas para 08-fine-tuning-lora."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestLoRAInit(unittest.TestCase):
    def test_B_zero(self):
        A, B = main.lora_init(8, 16, rank=4)
        np.testing.assert_array_equal(B, np.zeros_like(B))

    def test_A_nonzero(self):
        A, B = main.lora_init(8, 16, rank=4)
        self.assertGreater(np.abs(A).sum(), 0)


class TestLoRAForward(unittest.TestCase):
    def test_init_no_change(self):
        # Inicialmente B=0, deberia ser == base
        x = np.random.default_rng(0).standard_normal((2, 8))
        W = np.random.default_rng(1).standard_normal((8, 16)) * 0.1
        A, B = main.lora_init(8, 16, rank=4)
        out = main.lora_forward(x, W, A, B)
        np.testing.assert_allclose(out, x @ W, atol=1e-9)

    def test_with_trained_B(self):
        x = np.random.default_rng(0).standard_normal((2, 8))
        W = np.zeros((8, 16))
        A, B = main.lora_init(8, 16, rank=4)
        B = np.ones((4, 16))
        out = main.lora_forward(x, W, A, B, rank=4, alpha=2.0)
        # (alpha/rank) * x @ A @ B = (2/4) * x @ A @ 1
        # Como B es ones: x @ A @ 1 = sum columns of A weighted by x
        # Solo verificamos que no es ceros
        self.assertGreater(np.abs(out).sum(), 0)


class TestLoRAMerge(unittest.TestCase):
    def test_basic(self):
        W = np.random.default_rng(0).standard_normal((4, 4))
        A = np.random.default_rng(1).standard_normal((4, 2))
        B = np.random.default_rng(2).standard_normal((2, 4))
        W_new = main.lora_merge(W, A, B, alpha=1.0, rank=2)
        # W + 0.5 * A @ B
        expected = W + 0.5 * (A @ B)
        np.testing.assert_allclose(W_new, expected, atol=1e-9)


class TestParams(unittest.TestCase):
    def test_lora_menos_que_full(self):
        in_dim, out_dim, rank = 64, 128, 8
        self.assertLess(main.lora_trainable_params(in_dim, out_dim, rank),
                        main.full_ft_params(in_dim, out_dim))


class TestQLoRA(unittest.TestCase):
    def test_config(self):
        c = main.qlora_config()
        self.assertTrue(c["load_in_4bit"])
        self.assertTrue(c["bnb_4bit_use_double_quant"])


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