"""Pruebas para 18-mcp-auth-production."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import time
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestJWT(unittest.TestCase):
    def setUp(self):
        self.secret = "test-secret"

    def test_make(self):
        token = main.make_jwt({"sub": "user_1"}, self.secret)
        parts = token.split(".")
        self.assertEqual(len(parts), 3)

    def test_verify_valid(self):
        token = main.make_jwt({"sub": "user_1", "scope": "tools:read"}, self.secret)
        ok, payload = main.verify_jwt(token, self.secret)
        self.assertTrue(ok)
        self.assertEqual(payload["sub"], "user_1")

    def test_verify_invalid_signature(self):
        token = main.make_jwt({"sub": "user_1"}, self.secret)
        ok, msg = main.verify_jwt(token, "wrong-secret")
        self.assertFalse(ok)

    def test_verify_expired(self):
        token = main.make_jwt({"sub": "x", "exp": int(time.time()) - 10}, self.secret)
        ok, msg = main.verify_jwt(token, self.secret)
        self.assertFalse(ok)
        self.assertIn("expired", msg)

    def test_verify_scope(self):
        token = main.make_jwt({"sub": "x", "scope": "tools:read"}, self.secret)
        ok, msg = main.verify_jwt(token, self.secret, required_scope="tools:read")
        self.assertTrue(ok)

    def test_verify_missing_scope(self):
        token = main.make_jwt({"sub": "x", "scope": "tools:read"}, self.secret)
        ok, msg = main.verify_jwt(token, self.secret, required_scope="tools:execute")
        self.assertFalse(ok)


class TestIssueToken(unittest.TestCase):
    def test_basic(self):
        token = main.issue_access_token("secret", "user_1", ["tools:read"])
        ok, payload = main.verify_jwt(token, "secret")
        self.assertTrue(ok)
        self.assertEqual(payload["scope"], "tools:read")


class TestRefresh(unittest.TestCase):
    def test_refresh(self):
        t = main.refresh_token("secret", "rt_x")
        self.assertIsNotNone(t)

    def test_rotate(self):
        t = main.rotate_refresh_token()
        self.assertTrue(t.startswith("rt_"))


class TestAuditLog(unittest.TestCase):
    def test_basic(self):
        log = main.audit_log("user_1", "tools/call", "foo", "ok")
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["actor"], "user_1")

    def test_appends(self):
        log = main.audit_log("a", "x", "t", "ok")
        log = main.audit_log("b", "y", "t", "fail", log=log)
        self.assertEqual(len(log), 2)


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