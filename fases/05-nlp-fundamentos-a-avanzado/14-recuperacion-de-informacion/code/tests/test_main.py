"""Pruebas para 14-recuperacion-de-informacion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestIndice(unittest.TestCase):
    def test_basico(self):
        docs = ["el gato come", "el perro ladra"]
        idx = main.construir_indice_invertido(docs)
        # 'gato' deberia estar en doc 0
        self.assertIn(0, idx["gato"])
        # 'perro' deberia estar en doc 1
        self.assertIn(1, idx["perro"])
        # 'come' deberia estar en doc 0
        self.assertIn(0, idx["come"])


class TestBM25(unittest.TestCase):
    def test_score_doc_relevante(self):
        docs = ["el gato come", "el perro ladra"]
        idx = main.construir_indice_invertido(docs)
        scores = main.bm25("gato", docs, idx)
        # Doc 0 deberia tener score > 0
        self.assertGreater(scores[0], 0)
        # Doc 1 deberia tener score 0
        self.assertEqual(scores[1], 0)


class TestPrecision(unittest.TestCase):
    def test_p1(self):
        self.assertEqual(main.precision_at_k([0], [0, 1, 2], 1), 1.0)
        self.assertEqual(main.precision_at_k([0], [1, 0, 2], 1), 0.0)


class TestMain(unittest.TestCase):
    def test_main_ejecuta(self):
        import io
        from contextlib import redirect_stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            rc = main.main()
        self.assertEqual(rc, 0)
        self.assertIn("Scores", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()