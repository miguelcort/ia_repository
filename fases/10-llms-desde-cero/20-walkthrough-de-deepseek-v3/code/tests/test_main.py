"""Pruebas para 20-walkthrough-de-deepseek-v3."""
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
        a = main.deepseek_v3_architecture()
        self.assertEqual(len(a), 15)
        self.assertEqual(a["Total params"], "671B")
        self.assertIn("MLA", a["Attention"])


class TestCosts(unittest.TestCase):
    def test_keys(self):
        c = main.deepseek_v3_costs()
        self.assertIn("Training tokens", c)
        self.assertEqual(c["Training tokens"], "14.8T")


class TestMoE(unittest.TestCase):
    def test_active(self):
        a = main.moe_routing(top_k=8, n_experts=256, n_shared=2)
        # 8 + 2 = 10
        self.assertEqual(a, 10)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)


class TestInnovations(unittest.TestCase):
    def test_innovations_list(self):
        """DeepSeek-V3 innovations."""
        innovations = main.deepseek_v3_innovations()
        self.assertIsInstance(innovations, list)
        self.assertGreater(len(innovations), 0)


if __name__ == "__main__":
    unittest.main()