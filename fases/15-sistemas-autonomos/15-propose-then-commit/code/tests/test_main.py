"""Pruebas para 15-propose-then-commit."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestProposeThenCommit(unittest.TestCase):
    def setUp(self):
        self.ptc = main.ProposeThenCommit()

    def test_propose(self):
        pid = self.ptc.propose("write", {"path": "x.txt"}, "Write x")
        self.assertIn(pid, self.ptc.pending)
        p = self.ptc.pending[pid]
        self.assertEqual(p["action"], "write")
        self.assertEqual(p["summary"], "Write x")

    def test_approve(self):
        pid = self.ptc.propose("write", {})
        p = self.ptc.approve(pid, approver="alice")
        self.assertEqual(p["action"], "write")
        self.assertNotIn(pid, self.ptc.pending)
        self.assertIn(pid, self.ptc.committed)
        self.assertEqual(self.ptc.committed[pid]["approver"], "alice")

    def test_reject(self):
        pid = self.ptc.propose("write", {})
        p = self.ptc.reject(pid, reason="unsafe")
        self.assertEqual(self.ptc.rejected[pid]["reason"], "unsafe")
        self.assertNotIn(pid, self.ptc.pending)

    def test_approve_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.ptc.approve("nope")

    def test_list_pending(self):
        self.ptc.propose("a", {})
        self.ptc.propose("b", {})
        self.assertEqual(len(self.ptc.list_pending()), 2)

    def test_default_summary(self):
        pid = self.ptc.propose("write", {"a": "x" * 200})
        p = self.ptc.pending[pid]
        self.assertEqual(p["summary"], str({"a": "x" * 200})[:80])


class TestPlan(unittest.TestCase):
    def test_add_step(self):
        p = main.Plan()
        p.add_step("step1", "read", requires_approval=False)
        self.assertEqual(len(p.steps), 1)
        self.assertEqual(p.steps[0]["id"], 0)

    def test_describe(self):
        p = main.Plan()
        p.add_step("Read", "read", requires_approval=False)
        p.add_step("Write", "write", requires_approval=True)
        s = p.describe()
        self.assertIn("Plan:", s)
        self.assertIn("[A]", s)
        self.assertIn("Read", s)
        self.assertIn("Write", s)

    def test_needs_approval(self):
        p = main.Plan()
        p.add_step("a", "x", requires_approval=False)
        p.add_step("b", "y", requires_approval=True)
        p.add_step("c", "z", requires_approval=True)
        self.assertEqual(p.needs_approval_count(), 2)

    def test_auto_vs_approval(self):
        p = main.Plan()
        p.add_step("a", "x", requires_approval=False)
        p.add_step("b", "y", requires_approval=True)
        self.assertEqual(len(p.auto_steps()), 1)
        self.assertEqual(len(p.approval_steps()), 1)


class TestCommitWithRollback(unittest.TestCase):
    def test_success(self):
        ptc = main.ProposeThenCommit()
        plan = main.Plan()
        plan.add_step("step1", "noop", {}, requires_approval=True)
        calls = []
        ok, info = main.commit_with_rollback(plan, ptc, lambda a, args: calls.append(a))
        self.assertTrue(ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(ptc.committed), 1)

    def test_rollback_on_failure(self):
        ptc = main.ProposeThenCommit()
        plan = main.Plan()
        plan.add_step("step1", "ok", {}, requires_approval=False)
        plan.add_step("step2", "fail", {}, requires_approval=True)
        def exec_fn(a, args):
            if a == "fail":
                raise ValueError("boom")
        ok, info = main.commit_with_rollback(plan, ptc, exec_fn)
        self.assertFalse(ok)
        self.assertIn("error", info)


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