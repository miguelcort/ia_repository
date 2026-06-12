"""Pruebas para 05-supervisor-orchestrator-pattern."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestWorker(unittest.TestCase):
    def test_create(self):
        w = main.Worker(role="dev")
        self.assertEqual(w.role, "dev")
        self.assertEqual(w.results, [])

    def test_assign(self):
        w = main.Worker()
        r = w.assign({"action": "code"})
        self.assertEqual(r, "done:code")
        self.assertEqual(len(w.results), 1)


class TestSupervisor(unittest.TestCase):
    def setUp(self):
        self.s = main.Supervisor()
        self.s.register_worker(main.Worker(role="developer"))
        self.s.register_worker(main.Worker(role="tester"))
        self.s.register_worker(main.Worker(role="reviewer"))

    def test_register(self):
        self.assertIn("developer", self.s.workers)
        self.assertIn("tester", self.s.workers)
        self.assertIn("reviewer", self.s.workers)

    def test_dispatch_specific(self):
        r = self.s.dispatch({"action": "code"}, role="developer")
        self.assertEqual(r, "done:code")

    def test_dispatch_unknown_role(self):
        with self.assertRaises(ValueError):
            self.s.dispatch({"action": "x"}, role="unknown")

    def test_select_role_developer(self):
        self.assertEqual(self.s._select_role({"action": "implement"}), "developer")
        self.assertEqual(self.s._select_role({"action": "code"}), "developer")

    def test_select_role_tester(self):
        self.assertEqual(self.s._select_role({"action": "test"}), "tester")

    def test_select_role_reviewer(self):
        self.assertEqual(self.s._select_role({"action": "review"}), "reviewer")

    def test_select_role_default(self):
        self.assertEqual(self.s._select_role({"action": "foo"}), "worker")

    def test_run_plan(self):
        plan = [{"action": "implement"}, {"action": "test"}, {"action": "review"}]
        results = self.s.run_plan(plan)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.s.completed), 3)


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