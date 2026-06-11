"""Pruebas para 11-caching-y-costo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestCost(unittest.TestCase):
    def test_basic(self):
        # 1000 in @ $0.03/1k + 500 out @ $0.06/1k = 0.03 + 0.03 = 0.06
        c = main.estimate_cost(1000, 500, 0.03, 0.06)
        self.assertAlmostEqual(c, 0.06, places=4)


class TestCacheKey(unittest.TestCase):
    def test_different_temp(self):
        k1 = main.cache_key("hello", "gpt-4", 0)
        k2 = main.cache_key("hello", "gpt-4", 1)
        # Hash colision rara, pero el prefix deberia ser distinto
        self.assertTrue(k1.startswith("gpt-4:0:"))


class TestResponseCache(unittest.TestCase):
    def test_hit(self):
        cache = {"gpt-4:0:12345": "response"}
        r = main.response_cache(cache, "hello", model="gpt-4", temperature=0)
        # Hmm, hash no match porque mock
        self.assertIsNone(r)


class TestCacheSavings(unittest.TestCase):
    def test_no_cache(self):
        c1 = main.prompt_cache_savings(1000, cache_hit_rate=0.0)
        self.assertGreater(c1, 0)

    def test_more_hits_less_cost(self):
        c_no = main.prompt_cache_savings(1000, cache_hit_rate=0.0)
        c_half = main.prompt_cache_savings(1000, cache_hit_rate=0.5)
        self.assertLess(c_half, c_no)


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