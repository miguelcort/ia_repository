"""Pruebas para 26-failure-modes-agentic."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestFailureMode(unittest.TestCase):
    def test_basic(self):
        fm = main.FailureMode("test", "desc")
        self.assertEqual(fm.name, "test")
        self.assertEqual(fm.count, 0)

    def test_record(self):
        fm = main.FailureMode("test", "desc")
        fm.record()
        fm.record()
        self.assertEqual(fm.count, 2)


class TestAgentFailureMonitor(unittest.TestCase):
    def setUp(self):
        self.mon = main.AgentFailureMonitor()

    def test_initial(self):
        self.assertEqual(self.mon.total_failures(), 0)

    def test_record(self):
        self.mon.record_failure("tool_error")
        self.assertEqual(self.mon.total_failures(), 1)

    def test_multiple_modes(self):
        self.mon.record_failure("tool_error")
        self.mon.record_failure("hallucination")
        self.mon.record_failure("infinite_loop")
        self.assertEqual(self.mon.total_failures(), 3)

    def test_report(self):
        self.mon.record_failure("tool_error")
        report = self.mon.get_report()
        self.assertIn("tool_error", report)
        self.assertEqual(report["tool_error"]["count"], 1)
        self.assertEqual(report["tool_error"]["severity"], "high")

    def test_unknown_mode(self):
        # no error raised
        self.mon.record_failure("unknown_mode")
        self.assertEqual(self.mon.total_failures(), 0)


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