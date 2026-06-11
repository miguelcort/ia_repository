# MCP auth production

> MCP auth production: OAuth 2.1 server (PKCE, token endpoints) + JWT validation (HS256 shared secret hmac.compare_digest, RS256 public key, EdDSA modern; payload sub + scope + iat + exp; exp > now check, required scope in payload.scope) + scope enforcement (tools:read, tools:execute, resources:read) + refresh tokens + rotation (issue refresh + access, use refresh to get new access, issue new refresh + invalidate old, -token reuse, +security, +revocation, +detectable) + audit log (actor, action, target, result). +Secure, +Standardized, +Auditable, +Revocation. Variants: HS256 single service, RS256 multi-service public key, EdDSA modern, ES256, PS256. Frameworks: mcp, fastmcp, authlib, PyJWT, python-jose. +Production: standard 2024-25. +Use cases: agent, RAG, automation, MCP. Decision: single service -> HS256, multi-service -> RS256, modern -> EdDSA, production -> RS256. Trade-offs: JWT + standard, opaque + simple, rotation + security, no rotation -security. 2025: +MCP + A2A + native + auth.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/16
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar make_jwt y verify_jwt (HS256).
- Implementar issue_access_token.
- Implementar refresh_token y rotate_refresh_token.
- Implementar audit_log.
- Diagnosticar HS256 vs RS256 vs EdDSA.

## Constrúyelo

```python
def make_jwt(payload, secret):
    header = {"alg": "HS256", "typ": "JWT"}
    h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
    p_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    signing_input = f"{h_b64}.{p_b64}"
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    s_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{h_b64}.{p_b64}.{s_b64}"
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-auth-prod
fase: 13
leccion: 18
---

1. OAuth 2.1 server.
2. JWT validation.
3. Scope enforcement.
4. Refresh rotation.
5. Audit log.
```

## Ejercicios

1. **JWT**: usar PyJWT con
   HS256 y RS256.
2. **Refresh**: implementar
   rotation custom.
3. **Desafio**: MCP server
   con auth production.

## Lecturas recomendadas

- "RFC 7519: JSON Web Token" (IETF, 2015)
- "PyJWT Documentation" (https://pyjwt.readthedocs.io)
- "OAuth 2.0 Token Rotation" (authlib, 2024)
- "MCP Auth Production" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Auth Production]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).