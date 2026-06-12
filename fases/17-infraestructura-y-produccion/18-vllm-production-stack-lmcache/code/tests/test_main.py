"""Pruebas para 18-vllm-production-stack-lmcache."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestLMCache(unittest.TestCase):
    def test_put_get(self):
        c = main.LMCache(max_size_mb=100)
        c.put("k1", "v1", 50)
        self.assertEqual(c.get("k1"), "v1")

    def test_get_missing(self):
        c = main.LMCache()
        self.assertIsNone(c.get("missing"))

    def test_size(self):
        c = main.LMCache()
        c.put("k1", "v1", 50)
        c.put("k2", "v2", 30)
        self.assertEqual(c.size_mb(), 80)

    def test_count(self):
        c = main.LMCache()
        c.put("k1", "v1", 10)
        c.put("k2", "v2", 10)
        self.assertEqual(c.count(), 2)

    def test_eviction(self):
        c = main.LMCache(max_size_mb=100)
        c.put("k1", "v1", 60)
        c.put("k2", "v2", 60)
        self.assertLessEqual(c.size_mb(), 100)

    def test_hit_rate(self):
        c = main.LMCache(max_size_mb=1000)
        c.put("k1", "v1", 50)
        c.put("k2", "v2", 50)
        c.get("k1")
        c.get("missing")
        rate = c.hit_rate()
        self.assertEqual(rate, 0.5)


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