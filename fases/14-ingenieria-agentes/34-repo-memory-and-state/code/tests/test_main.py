"""Pruebas para 34-repo-memory-and-state."""
from __future__ import annotations
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRepoMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.mem = main.RepoMemory(self.tmp)

    def test_add_get(self):
        self.mem.add("a", 1)
        e = self.mem.get("a")
        self.assertEqual(e["value"], 1)

    def test_get_missing(self):
        self.assertIsNone(self.mem.get("nope"))

    def test_persistence(self):
        self.mem.add("a", 1)
        mem2 = main.RepoMemory(self.tmp)
        self.assertEqual(mem2.get("a")["value"], 1)

    def test_search_keyword(self):
        self.mem.add("lang", "python")
        self.mem.add("tool", "pytest")
        results = self.mem.search("python")
        self.assertEqual(len(results), 1)

    def test_search_tag(self):
        self.mem.add("a", 1, tags=["meta"])
        self.mem.add("b", 2, tags=["code"])
        results = self.mem.search("", tag="meta")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["key"], "a")

    def test_delete(self):
        self.mem.add("a", 1)
        self.assertTrue(self.mem.delete("a"))
        self.assertIsNone(self.mem.get("a"))
        self.assertFalse(self.mem.delete("a"))

    def test_ttl_expired(self):
        self.mem.add("short", "x", ttl=0.05)
        time.sleep(0.1)
        self.assertIsNone(self.mem.get("short"))

    def test_ttl_active(self):
        self.mem.add("long", "x", ttl=10)
        self.assertIsNotNone(self.mem.get("long"))

    def test_list_keys(self):
        self.mem.add("a", 1)
        self.mem.add("b", 2)
        self.assertEqual(set(self.mem.list_keys()), {"a", "b"})


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