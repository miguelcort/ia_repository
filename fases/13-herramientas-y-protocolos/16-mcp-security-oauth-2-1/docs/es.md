# MCP security OAuth 2.1

> MCP security OAuth 2.1 (RFC 6749 + 7636): PKCE for public clients (code_verifier 43-128 chars alphanumeric + -._~, code_challenge = S256(verifier) base64url-encoded), authorization code flow, access tokens + refresh tokens, scopes (tools:read list, tools:execute call, resources:read, prompts:get, sampling:create), Bearer tokens. +Secure, +Standardized, +No shared secret, +Granular permissions, +Least privilege. Variants: auth code + PKCE, client credentials (server-to-server), device code. Flow: (1) generate code_verifier + code_challenge, (2) build auth URL con challenge, (3) user approves, (4) exchange code + verifier for tokens, (5) use Bearer token en headers. Frameworks: mcp, fastmcp, anthropic, authlib, requests-oauthlib. +Production: standard 2024-25. +Use cases: agent, RAG, automation, MCP. Decision: user-facing -> OAuth 2.1, server-to-server -> mTLS o OAuth client creds, scripting -> API key, production -> OAuth 2.1. Trade-offs: PKCE + secure, no PKCE -secure, granular + security, broad -security. 2025: +MCP + A2A + native + auth.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07, 13/15
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar generate_pkce_pair.
- Implementar build_authorization_url.
- Implementar exchange_code_for_token.
- Implementar validate_token y auth_headers.
- Diagnosticar scopes MCP.
- Diagnosticar OAuth 2.1 vs API key vs mTLS.

## Constrúyelo

```python
def generate_pkce_pair():
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return code_verifier, code_challenge
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-oauth
fase: 13
leccion: 16
---

1. PKCE.
2. Auth code flow.
3. Bearer tokens.
4. Scopes.
5. +Production.
```

## Ejercicios

1. **OAuth 2.1**: usar authlib
   con MCP server.
2. **Scopes**: implementar
   scope check.
3. **Desafio**: MCP server
   con OAuth 2.1.

## Lecturas recomendadas

- "RFC 6749: OAuth 2.0" (IETF, 2012)
- "RFC 7636: PKCE" (IETF, 2015)
- "OAuth 2.1: The Next Generation" (Aaron Parecki, 2023)
- "MCP Security Specification" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Security OAuth 2.1]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).