"""Pruebas para 24-multimodal-rag-cross-modal."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestEncode(unittest.TestCase):
    def test_text(self):
        emb = main.encode_text("hello", embed_dim=128)
        self.assertEqual(emb.shape, (128,))

    def test_image(self):
        img = np.random.default_rng(0).standard_normal((32, 32, 3))
        emb = main.encode_image(img, embed_dim=128)
        self.assertEqual(emb.shape, (128,))


class TestCosine(unittest.TestCase):
    def test_self(self):
        a = np.array([1.0, 0.0])
        sim = main.cosine_sim(a, a)
        self.assertAlmostEqual(sim, 1.0, places=6)


class TestMultimodalIndex(unittest.TestCase):
    def test_basic(self):
        rng = np.random.default_rng(0)
        docs = [
            {"type": "text", "content": "hello"},
            {"type": "image", "content": rng.standard_normal((32, 32, 3))},
            {"type": "text", "content": "world"},
        ]
        emb, meta = main.multimodal_index(docs)
        self.assertEqual(emb.shape, (3, 512))
        self.assertEqual(len(meta), 3)


class TestSearch(unittest.TestCase):
    def test_text_query(self):
        rng = np.random.default_rng(0)
        docs = [
            {"type": "text", "content": "hello world"},
            {"type": "image", "content": rng.standard_normal((32, 32, 3))},
        ]
        emb, meta = main.multimodal_index(docs)
        query = {"type": "text", "content": "hello"}
        results = main.multimodal_search(emb, meta, query, top_k=1)
        self.assertEqual(len(results), 1)


class TestHybrid(unittest.TestCase):
    def test_basic(self):
        rng = np.random.default_rng(0)
        docs = [
            {"type": "text", "content": "cat dog"},
            {"type": "text", "content": "fish"},
        ]
        emb, meta = main.multimodal_index(docs)
        query = {"type": "text", "content": "cat"}
        results = main.hybrid_search(emb, meta, query, keyword_index=None, top_k=1)
        self.assertEqual(len(results), 1)


class TestRerank(unittest.TestCase):
    def test_basic(self):
        results = [({"type": "text", "content": "a"}, 0.5),
                   ({"type": "text", "content": "b"}, 0.9)]
        def reranker(query, doc):
            return 0.1 if doc["content"] == "a" else 0.8
        out = main.rerank(results, None, reranker)
        self.assertEqual(out[0][0]["content"], "b")


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