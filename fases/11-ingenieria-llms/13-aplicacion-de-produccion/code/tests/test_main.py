"""Pruebas para 13-aplicacion-de-produccion."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestChunk(unittest.TestCase):
    def test_basic(self):
        text = "a b c d e f g h i j"
        chunks = main.chunk_response(text, chunk_size=3)
        self.assertEqual(len(chunks), 4)


class TestRetry(unittest.TestCase):
    def test_exponential(self):
        delays = main.retry_with_backoff(3, base_delay=1)
        self.assertEqual(delays, [1, 2, 4])


class TestTimeout(unittest.TestCase):
    def test_7b_simple(self):
        self.assertEqual(main.timeout_seconds("simple", "7B"), 30)
    def test_70b_long(self):
        self.assertEqual(main.timeout_seconds("long", "70B"), 600)


class TestRateLimit(unittest.TestCase):
    def test_under(self):
        self.assertTrue(main.rate_limit_check([1] * 50, max_per_minute=60))

    def test_over(self):
        self.assertFalse(main.rate_limit_check([1] * 100, max_per_minute=60))


class TestErrors(unittest.TestCase):
    def test_keys(self):
        e = main.error_responses()
        self.assertIn("rate_limit", e)
        self.assertEqual(e["rate_limit"]["status"], 429)


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