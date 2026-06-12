"""Pruebas para 12-durable-execution."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestWorkflowContext(unittest.TestCase):
    def test_step_runs_once(self):
        ctx = main.WorkflowContext()
        calls = []
        def f():
            calls.append(1)
            return 42
        a = ctx.step("a", f)
        b = ctx.step("a", f)
        self.assertEqual(a, 42)
        self.assertEqual(b, 42)
        self.assertEqual(len(calls), 1)

    def test_step_different_keys(self):
        ctx = main.WorkflowContext()
        ctx.step("a", lambda: 1)
        ctx.step("b", lambda: 2)
        self.assertEqual(ctx.state["a"], 1)
        self.assertEqual(ctx.state["b"], 2)

    def test_checkpoint_restore(self):
        ctx = main.WorkflowContext()
        ctx.step("x", lambda: 100)
        snap = ctx.checkpoint()
        ctx2 = main.WorkflowContext.restore(snap)
        self.assertEqual(ctx2.state["x"], 100)
        self.assertIn("x", ctx2.steps_completed)

    def test_run_id_persisted(self):
        ctx = main.WorkflowContext(run_id="test-123")
        snap = ctx.checkpoint()
        ctx2 = main.WorkflowContext.restore(snap)
        self.assertEqual(ctx2.run_id, "test-123")


class TestDecorators(unittest.TestCase):
    def test_is_durable_step(self):
        @main.is_durable_step
        def my_step():
            return 1
        self.assertTrue(getattr(my_step, "_durable", False))


class TestRetryPolicy(unittest.TestCase):
    def test_default(self):
        p = main.retry_policy()
        self.assertEqual(p["max_attempts"], 3)
        self.assertEqual(p["backoff"], 1.5)

    def test_custom(self):
        p = main.retry_policy(max_attempts=5, backoff=2.0, initial_delay=0.5)
        self.assertEqual(p["max_attempts"], 5)
        self.assertEqual(p["backoff"], 2.0)
        self.assertEqual(p["initial_delay"], 0.5)


class TestSleepUntilResumed(unittest.TestCase):
    def test_sleep(self):
        ctx = main.WorkflowContext()
        result = main.sleep_until_resumed(ctx, resume_at=123)
        self.assertEqual(result["status"], "sleeping")
        self.assertEqual(result["resume_at"], 123)
        self.assertEqual(result["ctx"]["run_id"], ctx.run_id)


class TestFrameworks(unittest.TestCase):
    def test_list(self):
        fws = main.list_durable_frameworks()
        self.assertIn("temporal", fws)
        self.assertIn("inngest", fws)
        self.assertIn("restate", fws)
        self.assertIn("aws_sfn", fws)
        self.assertIn("dbos", fws)
        self.assertIn("cadence", fws)

    def test_get(self):
        f = main.get_framework("temporal")
        self.assertEqual(f["vendor"], "Temporal")
        self.assertIn("workflows", f["features"])

    def test_filter_python(self):
        result = main.filter_by_language("python")
        self.assertIn("temporal", result)
        self.assertIn("restate", result)
        self.assertIn("dbos", result)

    def test_temporal_year(self):
        f = main.get_framework("temporal")
        self.assertEqual(f["year"], 2019)


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