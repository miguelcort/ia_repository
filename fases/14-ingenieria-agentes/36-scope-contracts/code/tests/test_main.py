"""Pruebas para 36-scope-contracts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestScopeContract(unittest.TestCase):
    def test_in_scope(self):
        c = main.ScopeContract("dev", ["read", "write"], ["delete"])
        ok, reason = c.allows("read file")
        self.assertTrue(ok)
        self.assertEqual(reason, "in_scope")

    def test_out_of_scope(self):
        c = main.ScopeContract("dev", ["read"], ["delete"])
        ok, reason = c.allows("delete file")
        self.assertFalse(ok)
        self.assertIn("out_of_scope", reason)

    def test_not_in_scope(self):
        c = main.ScopeContract("dev", ["read"], ["delete"])
        ok, reason = c.allows("send email")
        self.assertFalse(ok)
        self.assertEqual(reason, "not_in_scope")

    def test_case_insensitive(self):
        c = main.ScopeContract("dev", ["Read"], ["Delete"])
        ok, _ = c.allows("READ file")
        self.assertTrue(ok)


class TestScopeRegistry(unittest.TestCase):
    def test_first_match(self):
        r = main.ScopeRegistry()
        r.add(main.ScopeContract("a", ["read"], []))
        r.add(main.ScopeContract("b", ["write"], []))
        ok, name, reason = r.validate("read x")
        self.assertTrue(ok)
        self.assertEqual(name, "a")

    def test_no_match(self):
        r = main.ScopeRegistry()
        r.add(main.ScopeContract("a", ["read"], []))
        ok, name, reason = r.validate("send email")
        self.assertFalse(ok)
        self.assertEqual(reason, "no_contract_allows")

    def test_multiple(self):
        r = main.ScopeRegistry()
        r.add(main.ScopeContract("dev", ["read", "write"], []))
        r.add(main.ScopeContract("prod", ["monitor"], []))
        ok, name, _ = r.validate("write code")
        self.assertEqual(name, "dev")
        ok, name, _ = r.validate("monitor service")
        self.assertEqual(name, "prod")


class TestParseScope(unittest.TestCase):
    def test_comma_split(self):
        c = main.parse_scope("read, write, fix", "delete, deploy", "dev")
        self.assertEqual(len(c.in_scope), 3)
        self.assertEqual(len(c.out_of_scope), 2)

    def test_newline_split(self):
        c = main.parse_scope("read\nwrite", "delete", "x")
        self.assertEqual(len(c.in_scope), 2)

    def test_semicolon_split(self):
        c = main.parse_scope("a;b;c", "x;y", "z")
        self.assertEqual(len(c.in_scope), 3)

    def test_parsed_contract_works(self):
        c = main.parse_scope("read, write", "delete", "dev")
        ok, _ = c.allows("read file")
        self.assertTrue(ok)
        ok, _ = c.allows("delete file")
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