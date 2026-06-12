"""Pruebas para 38-verification-gates."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestGate(unittest.TestCase):
    def test_run_success(self):
        g = main.Gate("x", lambda: 42)
        r = g.run()
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"], 42)

    def test_run_failure(self):
        g = main.Gate("x", lambda: (_ for _ in ()).throw(ValueError("boom")))
        r = g.run()
        self.assertFalse(r["ok"])
        self.assertEqual(r["error"], "boom")


class TestGateSequence(unittest.TestCase):
    def test_add_run(self):
        gs = main.GateSequence()
        gs.add(main.Gate("a", lambda: 1))
        gs.add(main.Gate("b", lambda: 2))
        results = gs.run()
        self.assertEqual(len(results), 2)

    def test_fail_fast(self):
        gs = main.GateSequence(fail_fast=True)
        gs.add(main.Gate("a", lambda: 1))
        gs.add(main.Gate("b", lambda: (_ for _ in ()).throw(ValueError("x"))))
        gs.add(main.Gate("c", lambda: 3))
        results = gs.run()
        self.assertEqual(len(results), 2)
        self.assertFalse(results[1]["ok"])

    def test_continue(self):
        gs = main.GateSequence(fail_fast=False)
        gs.add(main.Gate("a", lambda: 1))
        gs.add(main.Gate("b", lambda: (_ for _ in ()).throw(ValueError("x"))))
        gs.add(main.Gate("c", lambda: 3))
        results = gs.run()
        self.assertEqual(len(results), 3)

    def test_all_passed(self):
        gs = main.GateSequence()
        gs.add(main.Gate("a", lambda: 1))
        gs.add(main.Gate("b", lambda: 2))
        gs.run()
        self.assertTrue(gs.all_passed())

    def test_all_passed_with_failure(self):
        gs = main.GateSequence(fail_fast=False)
        gs.add(main.Gate("a", lambda: 1))
        gs.add(main.Gate("b", lambda: (_ for _ in ()).throw(ValueError("x"))))
        gs.run()
        self.assertFalse(gs.all_passed())


class TestHelpers(unittest.TestCase):
    def test_unit_tests(self):
        r = main.run_unit_tests(".")
        self.assertTrue(r["passed"])

    def test_lint(self):
        r = main.lint_check(".")
        self.assertEqual(r["issues"], 0)

    def test_type_check(self):
        r = main.type_check(".")
        self.assertEqual(r["errors"], 0)

    def test_contract_check_ok(self):
        self.assertTrue(main.contract_check(int, 5))

    def test_contract_check_fail(self):
        with self.assertRaises(ValueError):
            main.contract_check(int, "string")

    def test_schema_validate_ok(self):
        main.schema_validate({"a": int, "b": str}, {"a": 1, "b": "x"})

    def test_schema_validate_missing(self):
        with self.assertRaises(ValueError):
            main.schema_validate({"a": int}, {"b": 1})

    def test_schema_validate_type(self):
        with self.assertRaises(ValueError):
            main.schema_validate({"a": int}, {"a": "x"})


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