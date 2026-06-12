"""Pruebas para 01-why-multi-agent."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestReasons(unittest.TestCase):
    def test_list(self):
        reasons = main.list_reasons()
        self.assertIn("parallel", reasons)
        self.assertIn("specialization", reasons)
        self.assertIn("robustness", reasons)
        self.assertIn("collective", reasons)
        self.assertIn("evaluation", reasons)

    def test_get(self):
        r = main.get_reason("parallel")
        self.assertIn("use_cases", r)
        self.assertIn("cost", r)


class TestDownsides(unittest.TestCase):
    def test_list(self):
        downs = main.list_downsides()
        self.assertIn("coordination", downs)
        self.assertIn("cost", downs)

    def test_get(self):
        d = main.get_downside("cost")
        self.assertIn("description", d)


class TestDecide(unittest.TestCase):
    def test_simple_task(self):
        ok, reason = main.decide_use_multi_agent(0.2, 0.5, 0.5)
        self.assertFalse(ok)
        self.assertEqual(reason, "task_too_simple")

    def test_low_parallel(self):
        ok, _ = main.decide_use_multi_agent(0.5, 0.1, 0.5)
        self.assertFalse(ok)

    def test_tight_latency(self):
        ok, _ = main.decide_use_multi_agent(0.5, 0.5, 0.05)
        self.assertFalse(ok)

    def test_high_complexity(self):
        ok, _ = main.decide_use_multi_agent(0.9, 0.8, 0.5)
        self.assertTrue(ok)

    def test_balanced(self):
        ok, _ = main.decide_use_multi_agent(0.5, 0.5, 0.5)
        self.assertTrue(ok)


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