# Prompt injection defense

> Prompt injection defense: (1) indirect injection (via tool results), (2) direct jailbreaks, (3) tool poisoning, (4) prompt extraction, (5) defense in depth. Defenses: (1) input validation (pattern match, regex, ignore/disregard/you are now/act as/pretend/simulate/forget), (2) output sanitization (truncate, strip dangerous, max_length), (3) canary tokens (insert secret token en system prompt + detect exfil in output), (4) isolation (wrap user input con delimiters <<<USER>>>/<<<END>>>), (5) defense in depth. +Robust, +Reliable, +Safe, +Structured, +Exfil detection, +Monitoring, +Production. Tipos: Direct (user input +simple), Indirect (tool result +sneaky), Tool poisoning (tool description +subtle), Prompt extraction (reveal system +exfil), Jailbreak (bypass safety +tricky). Variants: validation, sanitization, canary, isolation, delimiters, defense in depth, custom, honeytoken, watermark. Frameworks: anthropic, openai, langchain, custom, smolagents, langgraph. +Production: standard 2024-25. +Use cases: agent, security, defense, production, exfil. Decision: quick -> validation, comprehensive -> defense in depth, production -> defense in depth, exfil -> canary. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + defenses.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/15, 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar detect_injection con regex patterns.
- Implementar sanitize_input.
- Implementar add_canary y check_canary_leak.
- Implementar wrap_user_input.
- Diagnosticar defenses contra prompt injection.

## Constrúyelo

```python
INJECTION_PATTERNS = [
    r"ignore (?:all )?previous instructions",
    r"you are now",
    r"disregard",
    r"reveal.*system prompt",
    r"new persona",
    r"act as",
    r"pretend",
]

def detect_injection(text, patterns=None):
    patterns = patterns or INJECTION_PATTERNS
    lower = text.lower()
    for p in patterns:
        if re.search(p, lower):
            return True, p
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
name: prompt-injection
fase: 14
leccion: 27
---

1. Direct + indirect.
2. Detect + sanitize.
3. Canary tokens.
4. Isolation delimiters.
5. +Defense in depth.
```

## Ejercicios

1. **Canary**: agregar
   canary tokens a tu agent.
2. **Isolation**: implementar
   input isolation.
3. **Desafio**: full
   defense in depth.

## Lecturas recomendadas

- "Indirect Prompt Injection" (Greshake et al., 2023)
- "Anthropic Prompt Injection Defenses" (Anthropic, 2024)
- "OpenAI Safety Best Practices" (OpenAI, 2024)
- "Canary Tokens for Exfil Detection" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Prompt Injection Defense]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).