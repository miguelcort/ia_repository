"""Pruebas para 16-mcp-security-oauth-2-1."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
RAIZ = sys.path.copy()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main
sys.path[:] = RAIZ


class TestPKCE(unittest.TestCase):
    def test_pair(self):
        v, c = main.generate_pkce_pair()
        # verifier >= 43 chars
        self.assertGreaterEqual(len(v), 43)
        # challenge != verifier
        self.assertNotEqual(v, c)
        # verifier alphanumeric + -._~
        for ch in v:
            self.assertIn(ch, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~")


class TestAuthURL(unittest.TestCase):
    def test_basic(self):
        url = main.build_authorization_url(
            "https://auth.example.com/auth",
            "client",
            "https://app/cb",
            "tools:read",
            "state",
            "challenge",
        )
        self.assertIn("response_type=code", url)
        self.assertIn("client_id=client", url)
        self.assertIn("code_challenge=challenge", url)
        self.assertIn("code_challenge_method=S256", url)


class TestTokenExchange(unittest.TestCase):
    def test_basic(self):
        r = main.exchange_code_for_token(
            "https://auth/token",
            "code",
            "verifier",
            "client",
            "redirect",
        )
        self.assertIn("access_token", r)
        self.assertEqual(r["token_type"], "Bearer")


class TestValidate(unittest.TestCase):
    def test_valid(self):
        ok, _ = main.validate_token("at_abc", [])
        self.assertTrue(ok)

    def test_invalid(self):
        ok, msg = main.validate_token("invalid", [])
        self.assertFalse(ok)


class TestAuthHeaders(unittest.TestCase):
    def test_format(self):
        h = main.auth_headers("at_abc")
        self.assertEqual(h["Authorization"], "Bearer at_abc")


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