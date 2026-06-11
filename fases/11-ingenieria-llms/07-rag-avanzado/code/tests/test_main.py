"""Pruebas para 07-rag-avanzado."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestHyDE(unittest.TestCase):
    def test_basic(self):
        h = main.hyde_generate_hypothetical("Q?")
        self.assertIn("Hipotetica", h)
        self.assertIn("Q?", h)


class TestMultiQuery(unittest.TestCase):
    def test_three(self):
        q = main.multi_query_rewrite("AI", n=3)
        self.assertEqual(len(q), 3)


class TestRRF(unittest.TestCase):
    def test_basic(self):
        rankings = [
            [("a", 0.9), ("b", 0.8)],
            [("b", 0.85), ("a", 0.7)],
        ]
        fused = main.reciprocal_rank_fusion(rankings)
        # a y b deberian estar en top
        ids = [d for d, _ in fused]
        self.assertIn("a", ids)
        self.assertIn("b", ids)


class TestRerank(unittest.TestCase):
    def test_sorted(self):
        docs = ["a", "b", "c"]
        scores = [0.5, 0.9, 0.1]
        reranked = main.cross_encoder_rerank("q", docs, mock_scores=scores)
        # Top deberia ser 'b'
        self.assertEqual(reranked[0][0], "b")


class TestCRAG(unittest.TestCase):
    def test_relevant(self):
        score = main.crag_score_relevance("AI es importante", "AI es la simulacion de inteligencia")
        self.assertGreater(score, 0.5)


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