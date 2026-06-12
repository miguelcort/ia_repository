"""Pruebas para 04-vllm-serving-internals."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPagedAttention(unittest.TestCase):
    def test_allocate(self):
        pa = main.PagedAttention(block_size=16, num_blocks=10)
        self.assertTrue(pa.allocate("seq1", 32))
        self.assertEqual(pa.memory_used(), 32)

    def test_allocate_partial_block(self):
        pa = main.PagedAttention(block_size=16, num_blocks=10)
        self.assertTrue(pa.allocate("seq1", 17))
        self.assertEqual(pa.memory_used(), 32)

    def test_allocate_fail(self):
        pa = main.PagedAttention(block_size=16, num_blocks=2)
        self.assertFalse(pa.allocate("seq1", 100))

    def test_free(self):
        pa = main.PagedAttention(block_size=16, num_blocks=10)
        pa.allocate("seq1", 32)
        pa.free("seq1")
        self.assertEqual(pa.memory_used(), 0)

    def test_memory_total(self):
        pa = main.PagedAttention(block_size=16, num_blocks=10)
        self.assertEqual(pa.memory_total(), 160)


class TestContinuousBatcher(unittest.TestCase):
    def test_add(self):
        b = main.ContinuousBatcher(max_batch_size=2)
        b.add({"id": "r1", "max_tokens": 10})
        b.add({"id": "r2", "max_tokens": 10})
        self.assertFalse(b.add({"id": "r3", "max_tokens": 10}))

    def test_step(self):
        b = main.ContinuousBatcher()
        b.add({"id": "r1", "max_tokens": 2})
        b.add({"id": "r2", "max_tokens": 5})
        active = b.step()
        self.assertEqual(len(active), 2)
        b.step()
        self.assertEqual(b.active_count(), 1)


class TestPrefixCache(unittest.TestCase):
    def test_no_match(self):
        self.assertEqual(main.prefix_cache_hit("hello", {}), 0)

    def test_full_match(self):
        cache = {"hello": "state"}
        self.assertEqual(main.prefix_cache_hit("hello world", cache), 5)

    def test_partial_longer(self):
        cache = {"hello": "state", "hello world": "state2"}
        self.assertEqual(main.prefix_cache_hit("hello world!", cache), 11)


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