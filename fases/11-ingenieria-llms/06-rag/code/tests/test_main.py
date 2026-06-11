"""Pruebas para 06-rag."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestVectorStore(unittest.TestCase):
    def test_basic(self):
        docs = ["a", "b"]
        embeds = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        store = main.simple_vector_store(docs, embeds)
        self.assertEqual(len(store), 2)


class TestRetrieve(unittest.TestCase):
    def test_top_k(self):
        docs = ["a", "b", "c"]
        embeds = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.9, 0.1])]
        store = main.simple_vector_store(docs, embeds)
        q = np.array([0.95, 0.05])
        top = main.retrieve(q, store, top_k=2)
        # Top deberia ser 'a' o 'c' (cercanos), no 'b'
        self.assertNotEqual(top[0], "b")


class TestRAGPrompt(unittest.TestCase):
    def test_basic(self):
        p = main.build_rag_prompt("Q?", ["Doc1", "Doc2"])
        self.assertIn("Contexto:", p)
        self.assertIn("Doc1", p)
        self.assertIn("Q?", p)


class TestPipeline(unittest.TestCase):
    def test_pipeline(self):
        docs = ["a"]
        embeds = [np.array([1.0, 0.0])]
        store = main.simple_vector_store(docs, embeds)
        result = main.rag_pipeline("q", store, lambda x: np.array([0.9, 0.1]))
        self.assertIn("a", result)


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