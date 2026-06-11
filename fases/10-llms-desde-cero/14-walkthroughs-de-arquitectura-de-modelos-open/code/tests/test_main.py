"""Pruebas para 14-walkthroughs-de-arquitectura-de-modelos-open."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestLlama3(unittest.TestCase):
    def test_components(self):
        a = main.llama3_architecture()
        self.assertEqual(len(a), 8)
        self.assertEqual(a["Norm"], "RMSNorm (pre-norm)")


class TestMistral(unittest.TestCase):
    def test_components(self):
        a = main.mistral_architecture()
        self.assertEqual(len(a), 8)
        self.assertEqual(a["Sliding window"], "Window 4096")


class TestQwen(unittest.TestCase):
    def test_components(self):
        a = main.qwen_architecture()
        self.assertEqual(len(a), 7)


class TestPhi(unittest.TestCase):
    def test_components(self):
        a = main.phi_architecture()
        self.assertEqual(len(a), 7)


class TestGemma(unittest.TestCase):
    def test_components(self):
        a = main.gemma_architecture()
        self.assertEqual(len(a), 6)


class TestSmolLM(unittest.TestCase):
    def test_components(self):
        a = main.smollm_architecture()
        self.assertEqual(len(a), 6)


class TestComparison(unittest.TestCase):
    def test_table(self):
        t = main.comparison_table()
        self.assertEqual(len(t), 7)
        self.assertEqual(t[0][0], "Llama 3 8B")


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