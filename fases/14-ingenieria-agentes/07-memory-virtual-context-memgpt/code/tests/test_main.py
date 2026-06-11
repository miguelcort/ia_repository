"""Pruebas para 07-memory-virtual-context-memgpt."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestVirtualContext(unittest.TestCase):
    def setUp(self):
        self.mem = main.VirtualContextMemory(core_size=3, recall_size=5)

    def test_add_to_core(self):
        self.mem.add_to_core("a")
        self.assertEqual(len(self.mem.get_core()), 1)

    def test_page_out(self):
        for i in range(5):
            self.mem.add_to_core(f"item {i}")
        # core size 3, so 2 should be paged out
        self.assertEqual(len(self.mem.get_core()), 3)
        self.assertEqual(self.mem.get_archival_size(), 2)
        self.assertEqual(self.mem.page_out_count, 2)

    def test_page_in(self):
        for i in range(5):
            self.mem.add_to_core(f"item {i}")
        # item 0 and 1 should be in archival
        results = self.mem.page_in_from_archival("item 0", n=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "item 0")

    def test_search_archival(self):
        for i in range(5):
            self.mem.add_to_core(f"item {i}")
        # archival has [item 0, item 1]; core has [item 2, 3, 4]
        results = self.mem.search_archival("item 1", n=2)
        self.assertIn("item 1", results)
        # core items should not be in archival search
        self.assertNotIn("item 3", results)

    def test_recall(self):
        for i in range(5):
            self.mem.add_to_recall(f"recall {i}")
        # recall_size=5, so all 5 fit
        self.assertEqual(len(self.mem.get_recall()), 5)

    def test_recall_rolling(self):
        mem = main.VirtualContextMemory(core_size=3, recall_size=2)
        for i in range(5):
            mem.add_to_recall(f"r{i}")
        # only last 2
        self.assertEqual(len(mem.get_recall()), 2)


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