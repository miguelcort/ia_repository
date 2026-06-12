# Bounded self improvement

> Bounded self-improvement: (1) limits (max_modifications, +Safe), (2) sandboxing (+Isolated +Safe), (3) verification (eval old vs new + improvement + threshold +Safe +Validated +Reliable), (4) rollback (save old code + on failure revert +Recoverable +Standard +Reliable +Detectable). Safer alternative to recursive self-improvement. +Safe, +Reliable, +Standard, +Production, +Validated, +Detectable, +Recoverable, +Guarantees, +Concern, +Safety. Variants: bounded seminal, recursive, DGM (2025 +evolutionary +self-improving +open-ended), Godel, custom. Frameworks: custom, anthropic, openai, deepmind. +Production: standard 2024-25. +Use cases: production, safety, self-improving, research, alignment, rollback. Decision: production -> bounded, research -> recursive, open-ended -> DGM, simple -> custom, production -> bounded. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + bounded.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15/07
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar BoundedSelfImprover con max_modifications.
- Implementar verify_improvement.
- Implementar apply_modification con verify check.
- Implementar rollback.
- Diagnosticar bounded vs recursive vs DGM.

## Constrúyelo

```python
class BoundedSelfImprover:
    def apply_modification(self, old_code, new_code, eval_fn, min_improvement=0.0):
        if len(self.modifications) >= self.max_modifications:
            return {"status": "limit_reached", "applied": len(self.modifications)}
        if self.require_verification:
            verified, improvement = self.verify_improvement(old_code, new_code, eval_fn, min_improvement)
            if not verified:
                return {"status": "rejected", "improvement": improvement}
        self.modifications.append({"from": old_code, "to": new_code, "timestamp": time.time()})
        return {"status": "applied", "code": new_code, "improvement": improvement}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: bounded-self
fase: 15
leccion: 08
---

1. Limits + sandboxing.
2. Verification.
3. Rollback.
4. +Safe +Production.
5. +Guarantees.
```

## Ejercicios

1. **Bounded**: implementar
   custom BoundedSelfImprover.
2. **Verification**: probar
   verification logic.
3. **Desafio**: full
   bounded pipeline.

## Lecturas recomendadas

- "Bounded Self-Improvement" (Anthropic, 2024)
- "DGM: Darwin Godel Machine" (2025)
- "Production Self-Improving Systems" (OpenAI, 2024)
- "Self-Improvement Safety" (DeepMind, 2024)

---

> 📚 **Adaptación al español de la lección [Bounded Self Improvement]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).