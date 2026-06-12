# Recursive self improvement

> Recursive self-improvement: AI modifies its own code (iterative, +self-modifying +theoretical). Theoretical risks: (1) loss of control (+uncontrollable), (2) goal drift (+misaligned), (3) emergent goals (-intended), (4) alignment failure (+unaligned). Limits actuales: (1) bounded modifications (+safe), (2) sandboxing (+isolated +safe), (3) rollback (+recoverable +safe). +Safe, +Reliable, +Standard, +Production, +Concern, +Safety. Variants: recursive self-improvement seminal, bounded, DGM (2025 +evolutionary +self-improving +open-ended), Godel, custom. Frameworks: custom, deepmind, openai, antropic. +Production: standard 2024-25. +Use cases: research, self-improving, alignment, safety. Decision: theoretical -> recursive, safe -> bounded, open-ended -> DGM, production -> bounded. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + self-improving.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15/04
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar SelfImprovingAgent con max_self_modifications.
- Implementar self_modify con limit.
- Implementar should_rollback si performance dropped.
- Diagnosticar risks vs limits.

## Constrúyelo

```python
class SelfImprovingAgent:
    def self_modify(self, current_code, modification_fn):
        if len(self.modifications) >= self.max_self_modifications:
            return {"status": "limit_reached", "modifications": len(self.modifications)}
        new_code = modification_fn(current_code)
        self.modifications.append({"from": current_code, "to": new_code, "timestamp": time.time()})
        return {"status": "applied", "code": new_code}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: recursive-self
fase: 15
leccion: 07
---

1. Self-modifying.
2. Bounded + sandbox.
3. Rollback.
4. +Self-modifying.
5. +Theoretical.
```

## Ejercicios

1. **Self-modifying**:
   implementar custom agent.
2. **Rollback**: probar
   rollback logic.
3. **Desafio**: full
   self-improving system.

## Lecturas recomendadas

- "Recursive Self-Improvement" (Yudkowsky, 2013)
- "DGM" (2025)
- "Bounded Self-Improvement" (Anthropic, 2024)
- "Self-Modifying AI" (DeepMind, 2024)

---

> 📚 **Adaptación al español de la lección [Recursive Self Improvement]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).