"""
Lección: 18-mcp-auth-production
Fase: 13
MCP auth production: OAuth 2.1 server, JWT validation, scope
enforcement, refresh tokens, token rotation, audit log.
"""
from __future__ import annotations
import time
import hashlib
import hmac
import json
import base64


def make_jwt(payload, secret):
    """Mock JWT: header.payload.signature (HS256)."""
    header = {"alg": "HS256", "typ": "JWT"}
    h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    signing_input = f"{h_b64}.{p_b64}"
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    s_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{h_b64}.{p_b64}.{s_b64}"


def verify_jwt(token, secret, required_scope=None):
    """Verify JWT (HS256) + check scope."""
    try:
        h_b64, p_b64, s_b64 = token.split(".")
        signing_input = f"{h_b64}.{p_b64}"
        expected_sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        actual_sig = base64.urlsafe_b64decode(s_b64 + "==")
        if not hmac.compare_digest(expected_sig, actual_sig):
            return False, "invalid signature"
        payload = json.loads(base64.urlsafe_b64decode(p_b64 + "=="))
        # exp check
        if "exp" in payload and payload["exp"] < time.time():
            return False, "expired"
        # scope check
        if required_scope:
            scopes = payload.get("scope", "").split()
            if required_scope not in scopes:
                return False, f"missing scope: {required_scope}"
        return True, payload
    except Exception as e:
        return False, f"invalid token: {e}"


def issue_access_token(secret, user_id, scopes, expires_in=3600):
    """Issue access token JWT."""
    payload = {
        "sub": user_id,
        "scope": " ".join(scopes),
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_in,
    }
    return make_jwt(payload, secret)


def refresh_token(secret, refresh_token_value, expires_in=3600):
    """Mock refresh: issue new access token."""
    return issue_access_token(secret, "user_from_refresh", ["tools:read"], expires_in=expires_in)


def rotate_refresh_token():
    """Mock refresh token rotation."""
    return f"rt_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]}"


def audit_log(actor, action, target, result, log=None):
    """Append to audit log."""
    if log is None:
        log = []
    log.append({
        "timestamp": time.time(),
        "actor": actor,
        "action": action,
        "target": target,
        "result": result,
    })
    return log


def main() -> int:
    secret = "my-secret-123"
    token = issue_access_token(secret, "user_1", ["tools:read", "tools:execute"])
    print(f"Token: {token[:50]}...")
    ok, payload = verify_jwt(token, secret, required_scope="tools:read")
    print(f"Verified: {ok}, payload sub: {payload['sub']}")
    log = audit_log("user_1", "tools/call", "get_weather", "ok")
    print(f"Audit log entries: {len(log)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())