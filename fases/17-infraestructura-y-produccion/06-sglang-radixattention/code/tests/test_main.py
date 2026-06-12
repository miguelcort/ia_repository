"""Pruebas para 06-sglang-radixattention."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRadixNode(unittest.TestCase):
    def test_create(self):
        n = main.RadixNode(key="x")
        self.assertEqual(n.key, "x")
        self.assertTrue(n.is_leaf)


class TestRadixTree(unittest.TestCase):
    def test_insert_lookup(self):
        t = main.RadixTree()
        t.insert(["a", "b", "c"], value="state")
        matched = t.lookup(["a", "b", "c", "d"])
        self.assertEqual(matched, ["a", "b", "c"])

    def test_lookup_no_match(self):
        t = main.RadixTree()
        matched = t.lookup(["x"])
        self.assertEqual(matched, [])

    def test_insert_multiple(self):
        t = main.RadixTree()
        t.insert(["a", "b"])
        t.insert(["a", "c"])
        matched = t.lookup(["a", "b"])
        self.assertEqual(matched, ["a", "b"])

    def test_size(self):
        t = main.RadixTree()
        t.insert(["a"])
        t.insert(["b"])
        self.assertEqual(t.size, 2)

    def test_evict_lru(self):
        t = main.RadixTree()
        t.insert(["old"], value="state")
        evicted = t.evict_lru(current_time=10**10)
        self.assertGreaterEqual(evicted, 1)


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