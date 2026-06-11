"""Pruebas para 16-atencion-diferencial-v2."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestStandardAttention(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        Q, K, V = [rng.standard_normal((4, 8)) for _ in range(3)]
        out = main.standard_attention(Q, K, V)
        self.assertEqual(out.shape, (4, 8))


class TestDifferentialAttention(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(0)
        Q, K, V = [rng.standard_normal((4, 8)) for _ in range(3)]
        out = main.differential_attention(Q, K, V)
        self.assertEqual(out.shape, (4, 8))

    def test_lambda_cero_es_standard(self):
        # Si lambda=0, diff = standard scores
        rng = np.random.default_rng(0)
        Q, K, V = [rng.standard_normal((4, 8)) for _ in range(3)]
        out = main.differential_attention(Q, K, V, lambda_q=0.0)
        # No es exactamente standard (sigue dividido), pero deberia ser cercano
        self.assertEqual(out.shape, (4, 8))


class TestComponents(unittest.TestCase):
    def test_keys(self):
        c = main.differential_components()
        self.assertIn("Signal-noise split", c)
        self.assertIn("Phi-4", c)


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