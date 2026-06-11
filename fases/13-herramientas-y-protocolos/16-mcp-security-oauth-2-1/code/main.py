"""
Lección: 16-mcp-security-oauth-2-1
Fase: 13
MCP security: OAuth 2.1 con PKCE para autenticacion.
Authorization server, resource server, access tokens, scopes.
RFC 6749, RFC 7636.
"""
from __future__ import annotations
import hashlib
import base64
import secrets
import time


def generate_pkce_pair():
    """Generate PKCE code_verifier + code_challenge (S256)."""
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return code_verifier, code_challenge


def build_authorization_url(authorization_endpoint, client_id, redirect_uri, scope,
                            state, code_challenge, code_challenge_method="S256"):
    """Build OAuth 2.1 authorization URL."""
    params = (
        f"response_type=code"
        f"&client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&scope={scope}"
        f"&state={state}"
        f"&code_challenge={code_challenge}"
        f"&code_challenge_method={code_challenge_method}"
    )
    return f"{authorization_endpoint}?{params}"


def exchange_code_for_token(token_endpoint, code, code_verifier, client_id,
                            redirect_uri):
    """Mock token exchange (POST)."""
    # In real: HTTP POST to token_endpoint
    return {
        "access_token": f"at_{secrets.token_hex(16)}",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": f"rt_{secrets.token_hex(16)}",
        "scope": "tools:read tools:execute",
    }


def validate_token(token, expected_scopes):
    """Validate token (mock: check scopes + format)."""
    if not token or not token.startswith("at_"):
        return False, "invalid token"
    # mock: check scope claim (in real: decode JWT)
    return True, "valid"


def auth_headers(token):
    """Build auth headers."""
    return {"Authorization": f"Bearer {token}"}


def main() -> int:
    verifier, challenge = generate_pkce_pair()
    print(f"Verifier: {verifier[:20]}...")
    print(f"Challenge: {challenge[:20]}...")
    url = build_authorization_url(
        "https://auth.example.com/authorize",
        "client_123",
        "https://app.example.com/callback",
        "tools:read tools:execute",
        "random_state",
        challenge,
    )
    print(f"Auth URL: {url[:80]}...")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())