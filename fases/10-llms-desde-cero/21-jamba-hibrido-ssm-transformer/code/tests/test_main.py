"""Pruebas para 21-jamba-hibrido-ssm-transformer."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestArchitecture(unittest.TestCase):
    def test_components(self):
        a = main.jamba_architecture()
        self.assertEqual(len(a), 9)
        self.assertEqual(a["Context"], "256K tokens")


class TestRatios(unittest.TestCase):
    def test_three(self):
        r = main.hybrid_architecture_ratios()
        self.assertEqual(len(r), 3)


class TestComplexity(unittest.TestCase):
    def test_ssm(self):
        c, m = main.ssm_complexity(seq_len=1000, dim=64, state_size=16)
        # 1000 * 64 * 16 = 1024000
        self.assertEqual(c, 1024000)
        # 64 * 16 = 1024
        self.assertEqual(m, 1024)

    def test_ssm_menor_que_attention(self):
        n = 1024
        ssm_c, _ = main.ssm_complexity(n, dim=64)
        attn_c = main.attention_complexity(n)
        # SSM O(n * dim * state), attn O(n^2)
        # n=1024, dim=64, state=16: ssm = 1024*64*16 = 1048576
        # attn = 1024^2 = 1048576 -> iguales
        # En n mas grande SSM gana
        n2 = 128000
        ssm_c, _ = main.ssm_complexity(n2, dim=4096)
        attn_c = main.attention_complexity(n2)
        self.assertLess(ssm_c, attn_c)


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