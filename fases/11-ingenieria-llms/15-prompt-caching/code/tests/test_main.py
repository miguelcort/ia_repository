"""Pruebas para 15-prompt-caching."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestKey(unittest.TestCase):
    def test_different(self):
        k1 = main.cache_key_from_prompt("a", "gpt-4", 0)
        k2 = main.cache_key_from_prompt("b", "gpt-4", 0)
        self.assertNotEqual(k1, k2)


class TestShouldCache(unittest.TestCase):
    def test_long(self):
        self.assertTrue(main.should_cache("a" * 300))

    def test_short(self):
        self.assertFalse(main.should_cache("short"))


class TestHitRate(unittest.TestCase):
    def test_basic(self):
        h = main.cache_hit_rate([1] * 100, n_unique=20)
        self.assertEqual(h, 0.2)


class TestTTL(unittest.TestCase):
    def test_response(self):
        self.assertEqual(main.ttl_seconds("response"), 3600)
    def test_provider(self):
        self.assertEqual(main.ttl_seconds("provider"), 300)


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