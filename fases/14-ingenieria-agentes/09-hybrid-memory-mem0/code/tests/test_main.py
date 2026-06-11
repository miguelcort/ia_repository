"""Pruebas para 09-hybrid-memory-mem0."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestCosineSim(unittest.TestCase):
    def test_identical(self):
        a = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(main.cosine_sim(a, a), 1.0, places=6)

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(main.cosine_sim(a, b), 0.0, places=6)

    def test_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        self.assertEqual(main.cosine_sim(a, b), 0.0)


class TestMem0Memory(unittest.TestCase):
    def setUp(self):
        self.mem = main.Mem0Memory(embed_dim=32)

    def test_add(self):
        mid = self.mem.add("test memory")
        self.assertEqual(mid, 0)
        self.assertEqual(len(self.mem.get_all()), 1)

    def test_search(self):
        self.mem.add("Python programming")
        self.mem.add("Cooking recipes")
        results = self.mem.search("Python", n=1)
        self.assertIn("Python", results[0]["content"])

    def test_short_term(self):
        for i in range(7):
            self.mem.add(f"item {i}")
        # short_term_max=5
        self.assertEqual(len(self.mem.get_short_term()), 5)

    def test_short_term_rolling(self):
        for i in range(3):
            self.mem.add(f"item {i}")
        st = self.mem.get_short_term()
        # last 3 are present
        contents = [m["content"] for m in st]
        self.assertIn("item 0", contents)
        self.assertIn("item 2", contents)


class TestGraph(unittest.TestCase):
    def test_add(self):
        result = main.add_to_graph(
            memory=1,
            entities=[{"name": "Alice"}, {"name": "Bob"}],
            relations=[{"from": "Alice", "to": "Bob", "type": "knows"}],
        )
        self.assertEqual(len(result["entities"]), 2)
        self.assertEqual(len(result["relations"]), 1)
        self.assertEqual(result["memory_id"], 1)


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