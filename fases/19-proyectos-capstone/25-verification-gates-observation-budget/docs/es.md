# 25 — Verification gates y observation budget

> Verification gates: el agent verifica resultados antes de proceder. Observation budget: limita output size (truncate, summarize). Previene context poisoning y cost runaway. Frameworks: guardrails-ai, custom validators.

**Tipo:** Construir
**Lenguajes:** Python, TypeScript
**Prerrequisitos:** Fase 14, Fase 19/20
**Tiempo estimado:** ~25 minutos

## Objetivos

- Implementar verification gates.
- Observation truncation.
- Cost tracking per turn.
- Context poisoning prevention.

## El problema

Verification gates: antes de aceptar un tool result,
validar (1) schema, (2) expected fields, (3) no
poisoning. Observation budget: truncar outputs grandes
(>8K tokens) a (8K tokens max). Cost tracking: por
turn, cumulative. Si excede budget → stop. Context
poisoning: el LLM puede ser "envenenado" por outputs
maliciosos (indirect prompt injection). Defenses:
(1) Tag data según trust level. (2) Strip
instrucciones de tool output. (3) Cap output size.

## Constrúyelo

```python
def verification_gate(tool_output, expected_schema,
                     trust_level):
    """Verificar output antes de pasar al LLM."""
    if not validate(tool_output, expected_schema):
        return {"status": "fail", "reason": "schema"}
    if trust_level == "untrusted":
        # Strip potential prompt injection
        tool_output = strip_injections(tool_output)
    # Truncate to observation budget
    truncated = truncate(tool_output, max_tokens=4096)
    return {"status": "ok", "output": truncated}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-verification
fase: 19
leccion: 25
---

1. Schema validation.
2. Trust level tagging.
3. Output truncation.
4. Cost tracking.
5. Poisoning prevention.
```

## Ejercicios

1. **Gate**: validate
   10 tool outputs.
2. **Truncate**: cap at
   4K tokens.
3. **Desafío**: detect
   indirect injection.

## Lecturas recomendadas

- "guardrails-ai" (2024)
- "BIPIA" (Yi 2023)
- "Microsoft Spotlighting"
  (2024)

---

> 📚 **Adaptación al español** de la lección
> "[25-verification-gates-observation-budget]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
