"""Pruebas para 11-cuantizacion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestAbmaxQuant(unittest.TestCase):
    def test_shape(self):
        x = np.random.default_rng(0).standard_normal((4, 4)) * 2
        x_q, scale = main.absmax_quantize(x, num_bits=8)
        self.assertEqual(x_q.shape, x.shape)

    def test_range(self):
        x = np.array([1.0, 2.0, -3.0])
        x_q, scale = main.absmax_quantize(x, num_bits=8)
        # INT8 range: -128 to 127
        self.assertGreaterEqual(x_q.min(), -128)
        self.assertLessEqual(x_q.max(), 127)

    def test_dequantize_roundtrip(self):
        x = np.array([1.0, 2.0, -3.0])
        x_q, scale = main.absmax_quantize(x, num_bits=8)
        x_dq = main.dequantize(x_q, scale)
        # Roundtrip deberia ser cerca
        np.testing.assert_allclose(x, x_dq, atol=0.1)


class TestNF4(unittest.TestCase):
    def test_shape(self):
        x = np.random.default_rng(0).standard_normal((4, 4))
        x_q, scale = main.nf4_quantize(x, num_bits=4)
        self.assertEqual(x_q.shape, x.shape)


class TestMemorySavings(unittest.TestCase):
    def test_int8(self):
        # 1GB -> 1GB (no savings)
        self.assertEqual(main.memory_savings(1024, num_bits=8), 1024)

    def test_int4(self):
        # 1GB -> 0.5GB
        self.assertEqual(main.memory_savings(1024, num_bits=4), 512)

    def test_int2(self):
        # 1GB -> 0.25GB
        self.assertEqual(main.memory_savings(1024, num_bits=2), 256)


class TestMethods(unittest.TestCase):
    def test_seis(self):
        m = main.quant_methods()
        self.assertEqual(len(m), 6)
        self.assertIn("INT8", m)
        self.assertIn("INT4", m)
        self.assertIn("NF4", m)


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