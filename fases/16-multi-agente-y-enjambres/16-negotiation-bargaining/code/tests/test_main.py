"""Pruebas para 16-negotiation-bargaining."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestZOPA(unittest.TestCase):
    def test_overlap(self):
        ok, zopa = main.zopa_check((0, 100), (50, 150))
        self.assertTrue(ok)
        self.assertEqual(zopa, (50, 100))

    def test_no_overlap(self):
        ok, zopa = main.zopa_check((0, 50), (60, 100))
        self.assertFalse(ok)
        self.assertIsNone(zopa)

    def test_exact_match(self):
        ok, zopa = main.zopa_check((0, 100), (0, 100))
        self.assertTrue(ok)


class TestNash(unittest.TestCase):
    def test_basic(self):
        result = main.nash_bargaining((0, 100), (0, 100))
        self.assertIsNotNone(result)

    def test_no_zopa(self):
        result = main.nash_bargaining((0, 50), (60, 100))
        self.assertIsNone(result)


class TestAlternating(unittest.TestCase):
    def test_basic(self):
        result = main.alternating_offers((0, 100), (50, 150), max_rounds=5)
        self.assertIsNotNone(result)

    def test_no_zopa(self):
        result = main.alternating_offers((0, 50), (60, 100), max_rounds=5)
        self.assertIsNone(result)


class TestTIOLI(unittest.TestCase):
    def test_in_range(self):
        self.assertTrue(main.take_it_or_leave_it(75, (0, 100)))

    def test_out_of_range(self):
        self.assertFalse(main.take_it_or_leave_it(150, (0, 100)))


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