# Inference platform economics

> Economics: (1) Per-token (input+output+model), (2) Batch (discount+async), (3) Reserved (discount+commit), (4) Spot (cheap+preemptible), (5) TCO (fixed+var+plan). InferencePlan: per_token_input+per_token_output+batch_discount+reserved_discount+cost(input, output, batch, reserved). tco: batch_ratio (0-1)+reserved (bool)+fixed_monthly (float)+non-batch+batch. Diferencias: (1) On-demand = full price+instant, (2) Batch = discount+async+24h SLA, (3) Reserved = commit+discount+predictable, (4) Spot = cheap+preemptible+best effort. Criterios: On-demand = instant+burst+pay per use, Batch = async+OK wait+discount, Reserved = predictable+commit+stable, Spot = cost-sensitive+best effort+preemptible. Decision: burst -> on-demand, async -> batch, predict -> reserved, cheap -> spot. Frameworks: openai, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + economics.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar InferencePlan con per-token + discounts.
- Implementar cost() con batch + reserved.
- Implementar tco() con batch_ratio + fixed_monthly.
- Diagnosticar pricing models.
- Diagnosticar on-demand vs batch vs reserved.

## Constrúyelo

```python
class InferencePlan:
    def cost(self, input_tokens, output_tokens, batch=False, reserved=False):
        c = self.per_token_input * input_tokens + self.per_token_output * output_tokens
        if batch:
            c *= (1 - self.batch_discount)
        if reserved:
            c *= (1 - self.reserved_discount)
        return c
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: inference-economics
fase: 17
leccion: 02
---

1. InferencePlan.
2. cost + tco.
3. Pricing models.
4. +Production.
```

## Ejercicios

1. **Plan**: probar
   cost + discounts.
2. **tco**: probar
   batch_ratio.
3. **Desafio**: integrar
   con OpenAI billing.

## Lecturas recomendadas

- "OpenAI Pricing" (OpenAI, 2024)
- "Cloud Economics" (Weinman, 2012)
- "TCO Analysis" (NIST, 2024)

---

> 📚 **Adaptación al español de la lección [Inference Platform Economics]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).