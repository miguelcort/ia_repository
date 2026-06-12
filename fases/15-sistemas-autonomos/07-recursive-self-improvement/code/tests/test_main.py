"""Pruebas para 07-recursive-self-improvement."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestSelfImproving(unittest.TestCase):
    def setUp(self):
        self.a = main.SelfImprovingAgent(max_self_modifications=3)

    def test_basic(self):
        self.assertEqual(self.a.name, "agent")
        self.assertEqual(self.a.max_self_modifications, 3)

    def test_self_modify(self):
        result = self.a.self_modify("v1", lambda c: "v2")
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["code"], "v2")
        self.assertEqual(len(self.a.modifications), 1)

    def test_max_limit(self):
        for i in range(3):
            self.a.self_modify("v1", lambda c: f"v{i+2}")
        result = self.a.self_modify("v1", lambda c: "v5")
        self.assertEqual(result["status"], "limit_reached")

    def test_evaluate(self):
        score = self.a.evaluate("code", lambda c: 0.9)
        self.assertEqual(score, 0.9)
        self.assertEqual(len(self.a.performance_history), 1)

    def test_rollback_check(self):
        # first eval
        self.a.evaluate("v1", lambda c: 0.5)
        # second eval lower
        self.a.evaluate("v2", lambda c: 0.3)
        self.assertTrue(self.a.should_rollback(0.3))

    def test_no_rollback(self):
        # first eval
        self.a.evaluate("v1", lambda c: 0.5)
        # second eval higher
        self.a.evaluate("v2", lambda c: 0.7)
        self.assertFalse(self.a.should_rollback(0.7))


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