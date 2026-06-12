"""Pruebas para 33-instructions-as-executable-constraints."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestConstraint(unittest.TestCase):
    def test_matches(self):
        c = main.Constraint("x", r"foo", "block")
        self.assertTrue(c.matches("this is foo bar"))
        self.assertFalse(c.matches("bar"))

    def test_check_ok(self):
        c = main.Constraint("x", r"foo", "block")
        ok, msg = c.check("bar")
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_check_violates(self):
        c = main.Constraint("x", r"foo", "block", "no foo allowed")
        ok, msg = c.check("foo bar")
        self.assertFalse(ok)
        self.assertEqual(msg, "no foo allowed")


class TestConstraintSet(unittest.TestCase):
    def test_add_list(self):
        cs = main.ConstraintSet()
        cs.add(main.Constraint("a", r"x", "block"))
        cs.add(main.Constraint("b", r"y", "block"))
        self.assertEqual(cs.list(), ["a", "b"])

    def test_validate_all_pass(self):
        cs = main.ConstraintSet([
            main.Constraint("a", r"foo", "block"),
        ])
        ok, v = cs.validate("bar")
        self.assertTrue(ok)
        self.assertEqual(v, [])

    def test_validate_one_fails(self):
        cs = main.ConstraintSet([
            main.Constraint("a", r"foo", "block"),
            main.Constraint("b", r"baz", "block"),
        ])
        ok, v = cs.validate("foo and baz")
        self.assertFalse(ok)
        self.assertEqual(len(v), 2)

    def test_block_callback(self):
        cs = main.ConstraintSet([
            main.Constraint("a", r"foo", "block"),
        ])
        captured = []
        ok = cs.block_on_violation("foo", lambda v: captured.extend(v))
        self.assertFalse(ok)
        self.assertEqual(len(captured), 1)


class TestParseInstructions(unittest.TestCase):
    def test_must_not(self):
        cs = main.parse_instructions("MUST NOT include PII")
        self.assertGreater(len(cs), 0)
        self.assertEqual(cs[0].action, "block")

    def test_must(self):
        cs = main.parse_instructions("MUST include greeting")
        self.assertGreater(len(cs), 0)
        self.assertEqual(cs[0].action, "require")

    def test_combined(self):
        cs = main.parse_instructions("MUST NOT include PII. MUST include greeting.")
        names = [c.action for c in cs]
        self.assertIn("block", names)
        self.assertIn("require", names)

    def test_enforced(self):
        cs = main.parse_instructions("MUST NOT include PII")
        cs_set = main.ConstraintSet(cs)
        ok, v = cs_set.validate("include PII now")
        self.assertFalse(ok)


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