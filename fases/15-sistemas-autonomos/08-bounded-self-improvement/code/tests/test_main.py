"""Pruebas para 08-bounded-self-improvement."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestBoundedSelfImprover(unittest.TestCase):
    def setUp(self):
        self.si = main.BoundedSelfImprover("test", max_modifications=3, require_verification=True)

    def test_basic(self):
        self.assertEqual(self.si.name, "test")
        self.assertEqual(self.si.max_modifications, 3)

    def test_apply_with_improvement(self):
        result = self.si.apply_modification("v1", "v2_longer", lambda c: len(c), min_improvement=2)
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["improvement"], 7)  # 9 - 2

    def test_apply_no_improvement(self):
        # same length -> no improvement
        result = self.si.apply_modification("v1", "v2", lambda c: len(c), min_improvement=1)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["improvement"], 0)

    def test_max_modifications(self):
        for i in range(3):
            self.si.apply_modification("v1", f"v{i+2}_longer", lambda c: len(c), min_improvement=0)
        result = self.si.apply_modification("v1", "v5", lambda c: len(c), min_improvement=0)
        self.assertEqual(result["status"], "limit_reached")

    def test_rollback(self):
        self.si.apply_modification("v1", "v2_longer", lambda c: len(c), min_improvement=0)
        old = self.si.rollback()
        self.assertEqual(old, "v1")
        self.assertEqual(len(self.si.modifications), 0)

    def test_rollback_empty(self):
        self.assertFalse(self.si.rollback())


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