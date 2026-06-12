# Managed LLM platforms

> Platforms: (1) OpenAI (gpt-4o+o1), (2) Anthropic (claude-3.5), (3) Google (gemini-1.5), (4) AWS Bedrock (multi-model), (5) Azure OpenAI (enterprise), (6) Together (open source), (7) Fireworks (fast OSS). Criteria: cheapest (Together+Fireworks), largest context (Gemini 1M), best quality (Claude+GPT-4), enterprise (Azure+Bedrock), open source (Together+Fireworks). Ventajas managed vs self-hosted: no ops (no GPU+no deploy), scale (auto+burst), latest models (frontier+updates), SLA (uptime+support), compliance (SOC2+HIPAA). Criterios: Managed = scale+no ops+latest, Self-hosted = cost+control+privacy, OSS = cost+control+custom. Decision: scale -> managed, cost -> self-host, control -> self-host, mix -> hybrid. Frameworks: openai, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + platforms.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Ninguno
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MANAGED_PLATFORMS con 7 platforms.
- Implementar list_platforms + get_platform.
- Implementar cheapest + largest_context.
- Implementar estimate_cost.
- Diagnosticar platforms criteria.

## Constrúyelo

```python
MANAGED_PLATFORMS = {
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "pricing_per_1m_input": 2.5,
        "pricing_per_1m_output": 10.0,
        "context_window": 128000,
    },
    # ... 7 platforms
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
name: managed-llm-platforms
fase: 17
leccion: 01
---

1. 7 platforms.
2. cheapest + largest.
3. estimate_cost.
4. +Production.
```

## Ejercicios

1. **Platforms**: probar
   los 7.
2. **Cost**: probar
   estimate.
3. **Desafio**: integrar
   con OpenAI SDK.

## Lecturas recomendadas

- "OpenAI API" (OpenAI, 2024)
- "Anthropic API" (Anthropic, 2024)
- "Vertex AI" (Google, 2024)

---

> 📚 **Adaptación al español de la lección [Managed LLM Platforms]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).