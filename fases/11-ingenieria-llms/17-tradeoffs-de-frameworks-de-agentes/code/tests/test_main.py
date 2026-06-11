"""Pruebas para 17-tradeoffs-de-frameworks-de-agentes."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestComparison(unittest.TestCase):
    def test_seis(self):
        c = main.framework_comparison()
        # Ajustado: deberia ser 7
        self.assertEqual(len(c), 7)
        self.assertIn("LangGraph", c)
        self.assertIn("DSPy", c)


class TestDecision(unittest.TestCase):
    def test_stateful(self):
        self.assertEqual(main.decision_matrix("stateful"), "LangGraph")

    def test_rag(self):
        self.assertEqual(main.decision_matrix("rag"), "LlamaIndex")

    def test_typescript(self):
        self.assertEqual(main.decision_matrix("typescript"), "Mastra")


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