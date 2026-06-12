"""Pruebas para 42-agent-workbench-capstone."""
from __future__ import annotations
import os
import sys
import tempfile
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestCapstone(unittest.TestCase):
    def test_inspect_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.py"), "w").close()
            with open(os.path.join(tmp, "b.py"), "w") as f:
                f.write("# TODO: thing\n")
            cb = main.CapstoneWorkbench(
                registry=None, memory=None,
                scope=main_scope_factory(), reviewer=main_reviewer_factory(),
                gate=main_gate_factory(),
                todos_fn=main_todos_factory(),
                stats_fn=main_stats_factory(),
            )
            result = cb.inspect_repo(tmp)
            self.assertIn("todos", result)
            self.assertIn("stats", result)
            self.assertEqual(len(result["todos"]), 1)

    def test_execute_in_scope(self):
        cb = main.CapstoneWorkbench(
            registry=None, memory=None,
            scope=main_scope_factory(), reviewer=main_reviewer_factory(),
            gate=main_gate_factory(),
            todos_fn=main_todos_factory(),
            stats_fn=main_stats_factory(),
        )
        result = cb.execute_action("read files")
        self.assertTrue(result["ok"])

    def test_execute_out_of_scope(self):
        cb = main.CapstoneWorkbench(
            registry=None, memory=None,
            scope=main_scope_factory(), reviewer=main_reviewer_factory(),
            gate=main_gate_factory(),
            todos_fn=main_todos_factory(),
            stats_fn=main_stats_factory(),
        )
        result = cb.execute_action("delete files")
        self.assertFalse(result["ok"])

    def test_review(self):
        cb = main.CapstoneWorkbench(
            registry=None, memory=None,
            scope=main_scope_factory(), reviewer=main_reviewer_factory(),
            gate=main_gate_factory(),
            todos_fn=main_todos_factory(),
            stats_fn=main_stats_factory(),
        )
        result = cb.review_output("hello world")
        self.assertTrue(result["approved"])

    def test_run_gates(self):
        cb = main.CapstoneWorkbench(
            registry=None, memory=None,
            scope=main_scope_factory(), reviewer=main_reviewer_factory(),
            gate=main_gate_factory(),
            todos_fn=main_todos_factory(),
            stats_fn=main_stats_factory(),
        )
        result = cb.run_gates(lambda: 42)
        self.assertTrue(result["ok"])

    def test_full_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "a.py"), "w").close()
            cb = main.CapstoneWorkbench(
                registry=None, memory=None,
                scope=main_scope_factory(), reviewer=main_reviewer_factory(),
                gate=main_gate_factory(),
                todos_fn=main_todos_factory(),
                stats_fn=main_stats_factory(),
            )
            result = cb.full_workflow(tmp, "read files", "hello world")
            self.assertEqual(result["step"], "done")
            self.assertTrue(result["ok"])

    def test_full_workflow_blocked(self):
        cb = main.CapstoneWorkbench(
            registry=None, memory=None,
            scope=main_scope_factory(), reviewer=main_reviewer_factory(),
            gate=main_gate_factory(),
            todos_fn=main_todos_factory(),
            stats_fn=main_stats_factory(),
        )
        result = cb.full_workflow(".", "delete files", "x")
        self.assertEqual(result["step"], "execute")
        self.assertFalse(result["ok"])


def main_scope_factory():
    from main_l36 import ScopeRegistry
    r = ScopeRegistry()
    r.add(ScopeRegistry_Contract("dev", ["read", "write"], ["delete"]))
    return r


def main_reviewer_factory():
    from main_l39 import ReviewChecklist, Reviewer
    cl = ReviewChecklist()
    cl.add("not_empty", lambda x: bool(x), weight=1.0)
    return Reviewer(cl, threshold=0.5)


def main_gate_factory():
    from main_l38 import Gate
    return Gate("noop", lambda: True)


def main_todos_factory():
    return lambda root: [{"path": "x", "line": 1, "pattern": "TODO", "text": "TODO: x"}]


def main_stats_factory():
    return lambda root: {"python": 5, "javascript": 2}


class ScopeRegistry_Contract:
    def __init__(self, name, in_scope, out_of_scope):
        self.name = name
        self.in_scope = [s.lower() for s in in_scope]
        self.out_of_scope = [s.lower() for s in out_of_scope]

    def allows(self, action):
        action_lower = action.lower()
        for forbidden in self.out_of_scope:
            if forbidden in action_lower:
                return False, f"out_of_scope: {forbidden}"
        for allowed in self.in_scope:
            if allowed in action_lower:
                return True, "in_scope"
        return False, "not_in_scope"


# Override the scope factory with local version
def main_scope_factory():
    r = _ScopeRegistry()
    r.add(ScopeRegistry_Contract("dev", ["read", "write"], ["delete"]))
    return r


class _ScopeRegistry:
    def __init__(self):
        self.contracts = []

    def add(self, c):
        self.contracts.append(c)

    def validate(self, action):
        for c in self.contracts:
            ok, reason = c.allows(action)
            if ok:
                return True, c.name, reason
        return False, None, "no_contract_allows"


def main_reviewer_factory():
    cl = _ReviewChecklist()
    cl.add("not_empty", lambda x: bool(x), weight=1.0)
    return _Reviewer(cl, threshold=0.5)


class _ReviewChecklist:
    def __init__(self):
        self.items = []

    def add(self, name, fn, weight=1.0):
        self.items.append({"name": name, "fn": fn, "weight": weight})


class _Reviewer:
    def __init__(self, checklist, threshold=0.7):
        self.checklist = checklist
        self.threshold = threshold

    def review(self, output):
        passed = []
        failed = []
        for item in self.checklist.items:
            ok = bool(item["fn"](output))
            if ok:
                passed.append(item["name"])
            else:
                failed.append(item["name"])
        total = sum(i["weight"] for i in self.checklist.items)
        pw = sum(i["weight"] for i in self.checklist.items if i["name"] in passed)
        score = pw / total if total > 0 else 0
        return {
            "approved": score >= self.threshold,
            "score": score,
            "passed": passed,
            "failed": failed,
        }


def main_gate_factory():
    return _Gate("noop", lambda *args, **kwargs: True)


class _Gate:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def run(self, *args, **kwargs):
        try:
            r = self.fn(*args, **kwargs)
            return {"ok": True, "result": r}
        except Exception as e:
            return {"ok": False, "error": str(e)}


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