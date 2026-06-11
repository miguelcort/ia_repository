"""Pruebas para 01-por-que-transformers."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestPathLength(unittest.TestCase):
    def test_rnn(self):
        # RNN: O(n)
        self.assertEqual(main.path_length_analysis(100, "rnn"), 100)

    def test_self_attention(self):
        # Self-attention: O(1)
        self.assertEqual(main.path_length_analysis(100, "self-attention"), 1)

    def test_cnn(self):
        # CNN: O(log_k(n))
        # log_3(64) = ~3.79, ceil = 4
        pl = main.path_length_analysis(64, "cnn")
        self.assertGreater(pl, 2)


class TestComplexity(unittest.TestCase):
    def test_rnn(self):
        # n * d^2
        self.assertEqual(main.compute_complexity(10, 5, "rnn"), 10 * 25)

    def test_self_attention(self):
        # n^2 * d
        self.assertEqual(main.compute_complexity(10, 5, "self-attention"), 100 * 5)

    def test_cnn(self):
        # 3 * n * d^2
        self.assertEqual(main.compute_complexity(10, 5, "cnn"), 3 * 10 * 25)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("self-attention", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()