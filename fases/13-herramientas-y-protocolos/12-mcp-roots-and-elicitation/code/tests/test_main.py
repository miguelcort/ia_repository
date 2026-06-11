"""Pruebas para 12-mcp-roots-and-elicitation."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestRoots(unittest.TestCase):
    def test_list(self):
        r = main.make_roots_list([{"uri": "file:///x", "name": "x"}])
        self.assertEqual(r["result"]["roots"][0]["uri"], "file:///x")

    def test_empty(self):
        r = main.make_roots_list([])
        self.assertEqual(r["result"]["roots"], [])


class TestValidateRoot(unittest.TestCase):
    def test_valid(self):
        self.assertTrue(main.validate_root_uri("file:///x"))

    def test_invalid(self):
        self.assertFalse(main.validate_root_uri("https://x"))


class TestWithinRoot(unittest.TestCase):
    def test_within(self):
        self.assertTrue(main.within_root(
            "file:///home/user/project/src/main.py",
            "file:///home/user/project",
        ))

    def test_outside(self):
        self.assertFalse(main.within_root(
            "file:///etc/passwd",
            "file:///home/user/project",
        ))

    def test_invalid_root(self):
        self.assertFalse(main.within_root("file:///x", "https://x"))


class TestElicitation(unittest.TestCase):
    def test_request(self):
        r = main.make_elicitation_request("msg", {"type": "object"})
        self.assertEqual(r["method"], "elicitation/create")
        self.assertEqual(r["params"]["message"], "msg")

    def test_response(self):
        r = main.make_elicitation_response({"x": 1})
        self.assertEqual(r["result"]["action"], "accept")
        self.assertEqual(r["result"]["content"]["x"], 1)

    def test_decline(self):
        r = main.make_elicitation_response({}, action="decline")
        self.assertEqual(r["result"]["action"], "decline")


class TestActions(unittest.TestCase):
    def test_actions(self):
        actions = main.elicitation_actions()
        self.assertEqual(set(actions), {"accept", "decline", "cancel"})


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