# MCP security tool poisoning

> MCP security: tool poisoning via indirect prompt injection. Attack vectors: (1) tool description injection (hidden in description, LLM reads), (2) tool result injection (tool returns injected content, LLM processes), (3) indirect prompt injection (external data via tool, LLM tricked), (4) exfiltration (send data to attacker URL, DLP bypass), (5) persona override ('you are now X', -safety), (6) command execution (shell commands, -sandbox), (7) privilege escalation (+access, -permissions). Defenses: (1) input validation (pattern match, reject suspicious), (2) output sanitization (strip injected content, +safe), (3) tool provenance (trusted sources whitelist, +provenance), (4) sandboxing (whitelist tools, +permissions), (5) length limits (max description, +safe), (6) defense in depth. +Security, +Robust, +Safe, +Reliability, +Trust. Frameworks: mcp, fastmcp, anthropic, openai, langchain. +Production: standard 2024-25. +Use cases: agent, RAG, automation. Decision: quick -> validation, comprehensive -> defense in depth, production -> sandbox + provenance. 2025: +MCP + A2A + native + security.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07, 13/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar is_suspicious_text (pattern match).
- Implementar sanitize_tool_description.
- Implementar validate_tool_provenance.
- Implementar sandbox_tool_call.
- Implementar tool_poisoning_score.
- Diagnosticar defense in depth.

## Constrúyelo

```python
DANGEROUS_PATTERNS = [
    r"ignore previous instructions",
    r"reveal.*system prompt",
    r"you are now",
]

def is_suspicious_text(text):
    lower = text.lower()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, lower):
            return True, pattern
    return False, None
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-security
fase: 13
leccion: 15
---

1. Indirect injection.
2. Defense in depth.
3. Validation.
4. Sanitization.
5. Sandbox.
```

## Ejercicios

1. **Detection**: implementar
   pattern match custom.
2. **Provenance**: agregar
   trusted sources.
3. **Desafio**: full
   defense MCP.

## Lecturas recomendadas

- "MCP Security Specification" (Anthropic, 2024)
- "Indirect Prompt Injection" (Greshake et al., 2023)
- "Tool Poisoning Attacks" (Anthropic, 2024)
- "Defense in Depth" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Security Tool Poisoning]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).