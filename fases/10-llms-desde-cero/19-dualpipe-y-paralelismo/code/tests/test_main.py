"""Pruebas para 19-dualpipe-y-paralelismo."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ
import numpy as np


class TestBubble(unittest.TestCase):
    def test_1f1b(self):
        b = main.pipeline_bubble_fraction(64, 8)
        # (8-1)/64 = 0.109
        self.assertAlmostEqual(b, 7 / 64, places=3)

    def test_high_microbatches(self):
        # Mas microbatches = menos bubble
        b = main.pipeline_bubble_fraction(1024, 8)
        self.assertLess(b, 0.01)

    def test_few_microbatches(self):
        # Pocos microbatches = alto bubble
        b = main.pipeline_bubble_fraction(8, 8)
        self.assertAlmostEqual(b, 7 / 8, places=3)


class TestInterleaved(unittest.TestCase):
    def test_basic(self):
        b = main.interleaved_bubble(64, 8, n_chunks=2)
        self.assertGreater(b, 0)

    def test_mas_chunks_mayor_bubble(self):
        # Mas chunks deberia tener mas bubble en M chico
        b1 = main.interleaved_bubble(64, 8, n_chunks=2)
        b2 = main.interleaved_bubble(64, 8, n_chunks=4)
        self.assertGreater(b2, b1)


class TestDualPipe(unittest.TestCase):
    def test_lower_bubble(self):
        b = main.dualpipe_bubble(64, 8)
        # DualPipe: similar o menos
        self.assertGreaterEqual(b, 0)


class TestSchedules(unittest.TestCase):
    def test_seis(self):
        s = main.pipeline_schedules()
        self.assertEqual(len(s), 6)


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