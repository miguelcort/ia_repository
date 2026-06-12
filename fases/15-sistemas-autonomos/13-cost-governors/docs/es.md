# Cost governors

> Cost governors: (1) Daily budget USD (cap+reset), (2) Per-request cap (max+reject), (3) Model routing (cheapest+quality), (4) Rate limits (RPM+TPM), (5) Concurrency caps (max+queue), (6) Circuit breakers (open+cooldown), (7) Soft + hard caps (warn+reject). CostGovernor mínimo: daily_budget_usd (float+reset), per_request_cap (float+reject), model_costs dict (USD/M+default), spend_log (list+append), estimate_cost = per_million * (in+out) / 1M, allow() check per_request+daily+append, spent() sum, remaining() budget - spent, reset() clear. Circuit breaker: failure_threshold (max+open), cooldown_seconds (wait+reset), failures (count+increment), opened_at (time+check), call(fn) check opened_at -> if open+in cooldown raise -> if open+past cooldown reset -> try fn -> catch: increment failures, open if threshold, is_open property. Criterios: cost = CostGovernor (budget, cap, routing), reliability = CircuitBreaker (failures, rate, concurrency), decision = daily cost -> CostGovernor, API failures -> CircuitBreaker, mix -> ambos, soft cap = warn, hard cap = reject, production = ambos. Frameworks: openai, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + cost.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar CostGovernor con daily_budget + per_request_cap.
- Implementar estimate_cost + allow + spent + remaining + reset.
- Implementar pick_cheapest_model.
- Implementar CircuitBreaker con threshold + cooldown.
- Diagnosticar governors.

## Constrúyelo

```python
class CostGovernor:
    def allow(self, model, input_tokens, output_tokens):
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        if cost > self.per_request_cap:
            return False, cost, "exceeds_per_request_cap"
        total = sum(c for _, c in self.spend_log)
        if total + cost > self.daily_budget_usd:
            return False, cost, "exceeds_daily_budget"
        self.spend_log.append((time.time(), cost))
        return True, cost, "ok"
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: cost-governors
fase: 15
leccion: 13
---

1. CostGovernor + CircuitBreaker.
2. Daily budget + per-request.
3. Model routing + soft/hard.
4. +Production.
```

## Ejercicios

1. **CostGovernor**: probar
   budget + cap.
2. **CircuitBreaker**: probar
   threshold + cooldown.
3. **Desafio**: integrar
   con OpenAI + Anthropic.

## Lecturas recomendadas

- "OpenAI: Rate Limits" (OpenAI, 2024)
- "Anthropic: Rate Limits" (Anthropic, 2024)
- "Circuit Breaker Pattern" (Nygard, 2007)

---

> 📚 **Adaptación al español de la lección [Cost Governors]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).