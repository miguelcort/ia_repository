"""Pruebas para 20-shadow-canary-progressive."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestStage(unittest.TestCase):
    def test_create(self):
        s = main.DeploymentStage("canary", 5)
        self.assertEqual(s.name, "canary")
        self.assertEqual(s.percent, 5)

    def test_record(self):
        s = main.DeploymentStage("canary", 5)
        s.record(error=False, latency=0.1)
        s.record(error=True, latency=0.2)
        self.assertEqual(s.metrics["requests"], 2)
        self.assertEqual(s.metrics["errors"], 1)

    def test_error_rate(self):
        s = main.DeploymentStage("canary", 5)
        s.record(error=True)
        s.record(error=False)
        self.assertEqual(s.error_rate(), 0.5)


class TestRollout(unittest.TestCase):
    def setUp(self):
        self.r = main.ProgressiveRollout()
        self.r.add_stage("canary", 5)
        self.r.add_stage("half", 50)
        self.r.add_stage("full", 100)

    def test_route_canary(self):
        s = self.r.route(0.03)
        self.assertEqual(s.name, "canary")

    def test_route_full(self):
        s = self.r.route(0.5)
        self.assertEqual(s.name, "full")

    def test_advance(self):
        s = self.r.advance()
        self.assertEqual(s.name, "half")

    def test_advance_end(self):
        self.r.advance()
        self.r.advance()
        s = self.r.advance()
        self.assertIsNone(s)

    def test_rollback(self):
        self.r.advance()
        self.r.advance()
        s = self.r.rollback()
        self.assertEqual(s.name, "canary")

    def test_shadow(self):
        result = self.r.shadow_test("req1", "new_resp", "old_resp")
        self.assertEqual(result, "old_resp")
        self.assertEqual(len(self.r.shadow_results), 1)
        self.assertTrue(self.r.shadow_results[0]["differ"])


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