"""Pruebas para 23-colpali-vision-native-rag."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestColBERT(unittest.TestCase):
    def test_late_interaction(self):
        rng = np.random.default_rng(0)
        doc = rng.standard_normal((10, 8))
        query = rng.standard_normal((3, 8))
        score = main.colbert_late_interaction(doc, query)
        self.assertIsInstance(score, float)
        # self-similarity is max
        score_self = main.colbert_late_interaction(doc, doc[:3])
        self.assertGreater(score_self, 0)


class TestEncode(unittest.TestCase):
    def test_page(self):
        img = np.random.default_rng(0).standard_normal((512, 512, 3))
        emb = main.encode_page_pali_gemma(img, n_patches=64, embed_dim=128)
        self.assertEqual(emb.shape, (64, 128))

    def test_query(self):
        emb = main.encode_query("test query", n_tokens=5, embed_dim=128)
        self.assertEqual(emb.shape, (5, 128))


class TestRetrieve(unittest.TestCase):
    def test_top_k(self):
        rng = np.random.default_rng(0)
        pages = [rng.standard_normal((128, 128, 3)) for _ in range(5)]
        retrieved = main.colpali_retrieve(pages, "test", top_k=3)
        self.assertEqual(len(retrieved), 3)
        for idx, score in retrieved:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 5)

    def test_top_k_larger_than_pages(self):
        rng = np.random.default_rng(0)
        pages = [rng.standard_normal((128, 128, 3)) for _ in range(2)]
        retrieved = main.colpali_retrieve(pages, "test", top_k=5)
        # at most 2 pages
        self.assertLessEqual(len(retrieved), 2)


class TestColpaliRAG(unittest.TestCase):
    def test_rag(self):
        rng = np.random.default_rng(0)
        pages = [rng.standard_normal((128, 128, 3)) for _ in range(3)]
        def gen(prompt):
            return "Mock answer based on context"
        out = main.colpali_rag(pages, "test", gen)
        self.assertIn("Mock answer", out)


class TestColQwen2(unittest.TestCase):
    def test_basic(self):
        rng = np.random.default_rng(0)
        pages = [rng.standard_normal((128, 128, 3)) for _ in range(3)]
        out = main.colqwen2_smol_vlm(pages, "test")
        self.assertEqual(len(out), 3)


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