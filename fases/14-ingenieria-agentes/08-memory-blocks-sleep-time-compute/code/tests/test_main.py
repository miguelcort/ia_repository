"""Pruebas para 08-memory-blocks-sleep-time-compute."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestMemoryBlock(unittest.TestCase):
    def test_basic(self):
        b = main.MemoryBlock("test", value="hello", limit=100)
        self.assertEqual(b.get(), "hello")

    def test_update(self):
        b = main.MemoryBlock("test", limit=100)
        b.update("new value")
        self.assertEqual(b.get(), "new value")

    def test_truncate(self):
        b = main.MemoryBlock("test", limit=5)
        b.update("a very long string")
        self.assertEqual(len(b.get()), 5)


class TestMemoryBlocksManager(unittest.TestCase):
    def setUp(self):
        self.mgr = main.MemoryBlocksManager()

    def test_add(self):
        self.mgr.add_block(main.MemoryBlock("a"))
        self.assertIn("a", self.mgr.all_blocks())

    def test_update(self):
        self.mgr.add_block(main.MemoryBlock("a", value="x"))
        self.mgr.update_block("a", "y")
        self.assertEqual(self.mgr.get_block("a").get(), "y")

    def test_total_size(self):
        self.mgr.add_block(main.MemoryBlock("a", value="abc"))
        self.mgr.add_block(main.MemoryBlock("b", value="de"))
        self.assertEqual(self.mgr.total_size(), 5)


class TestSleepTimeConsolidate(unittest.TestCase):
    def test_basic(self):
        mgr = main.MemoryBlocksManager()
        mgr.add_block(main.MemoryBlock("a", value="x" * 1000, limit=2000))
        consolidated = main.sleep_time_consolidate(mgr, main.summarize_memory)
        # value > 500 -> summarized
        self.assertEqual(len(consolidated), 1)
        self.assertLess(len(consolidated["a"]), 600)

    def test_no_change(self):
        mgr = main.MemoryBlocksManager()
        mgr.add_block(main.MemoryBlock("a", value="short"))
        consolidated = main.sleep_time_consolidate(mgr, main.summarize_memory)
        self.assertEqual(consolidated, {})


class TestBackgroundAgent(unittest.TestCase):
    def test_basic(self):
        mgr = main.MemoryBlocksManager()
        mgr.add_block(main.MemoryBlock("human", value="User is Alice"))
        results = main.background_agent(mgr, "Alice")
        self.assertIn("human", results)

    def test_no_match(self):
        mgr = main.MemoryBlocksManager()
        mgr.add_block(main.MemoryBlock("a", value="irrelevant"))
        results = main.background_agent(mgr, "specific_query")
        self.assertEqual(results, {})


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