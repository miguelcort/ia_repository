# Security secrets audit

> Security: (1) API keys (OpenAI/Anthropic), (2) Tokens (GitHub), (3) Passwords (in URLs), (4) Prompt injection (detect), (5) Prompt leakage (monitor). SecretsScanner: findings+scan (text) iterate SECRET_PATTERNS+iterate INJECTION+has_findings+secrets_count+injection_count. redact: re.sub patterns+[REDACTED]. SECRET_PATTERNS: openai_key+anthropic_key+aws_key+github_token+password_in_url+private_key+jwt_token. PROMPT_INJECTION_PATTERNS: ignore/disregard/forget instructions. Secrets: OpenAI/Anthropic (sk-/sk-ant-)+AWS (AKIA)+GitHub (ghp_)+Passwords in URLs+Private keys (RSA/EC/DSA/OPENSSH)+JWT (eyJ). Criterios: Gitleaks = simple+fast+Git, Custom = specific+tailored+domain, TruffleHog = deep+history+verifiers. Decision: simple -> gitleaks, specific -> custom, deep -> TruffleHog, mix -> all. Frameworks: gitleaks, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + security.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/24
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar SECRET_PATTERNS con 7 patterns.
- Implementar PROMPT_INJECTION_PATTERNS.
- Implementar SecretsScanner con scan + counts.
- Implementar redact function.
- Diagnosticar security types.

## Constrúyelo

```python
SECRET_PATTERNS = {
    "openai_key": r"sk-[A-Za-z0-9]{20,}",
    "anthropic_key": r"sk-ant-[A-Za-z0-9]{20,}",
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "github_token": r"ghp_[A-Za-z0-9]{36}",
    # ... 7 patterns
}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: security-secrets-audit
fase: 17
leccion: 25
---

1. SECRET_PATTERNS.
2. SecretsScanner.
3. redact.
4. +Production.
```

## Ejercicios

1. **Scanner**: probar
   detect.
2. **Redact**: probar
   substitution.
3. **Desafio**: integrar
   con gitleaks pre-commit.

## Lecturas recomendadas

- "Gitleaks" (Gitleaks, 2024)
- "TruffleHog" (TruffleHog, 2024)
- "Prompt Injection" (OWASP, 2024)

---

> 📚 **Adaptación al español de la lección [Security Secrets Audit]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).