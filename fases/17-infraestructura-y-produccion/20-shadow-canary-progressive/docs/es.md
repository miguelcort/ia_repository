# Shadow canary progressive

> Deploys: (1) Shadow (parallel+no user), (2) Canary (1% users+test), (3) Progressive (ramp up+5%->50%->100%), (4) Monitor (metrics), (5) Rollback (auto). DeploymentStage: name+percent+metrics+record (error, latency)+error_rate. ProgressiveRollout: stages list+current_stage int+shadow_results+add_stage+shadow_test (req, new, old)+route (user_hash)+advance+rollback. Diferencias: Shadow = parallel+no user+compare, Canary = 1% users+test+gradual, Blue-green = swap+two envs+cutover. Criterios: Shadow = compare+no user+pre-launch, Canary = gradual+small+production, Progressive = ramp+stages+control, Blue-green = swap+two envs+fast. Decision: compare -> shadow, gradual -> canary, ramp -> progressive, swap -> blue-green. Frameworks: argo, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + deploy.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/19
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar DeploymentStage con metrics.
- Implementar record + error_rate.
- Implementar ProgressiveRollout con stages.
- Implementar route + advance + rollback.
- Implementar shadow_test.
- Diagnosticar shadow vs canary vs blue-green.

## Constrúyelo

```python
def route(self, user_id_hash):
    if self.current_stage >= len(self.stages):
        return self.stages[-1] if self.stages else None
    if user_id_hash < self.stages[self.current_stage].percent:
        return self.stages[self.current_stage]
    return self.stages[-1] if self.stages else None
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: shadow-canary-progressive
fase: 17
leccion: 20
---

1. DeploymentStage.
2. ProgressiveRollout.
3. route + advance.
4. shadow_test.
5. +Production.
```

## Ejercicios

1. **Stage**: probar
   record + error_rate.
2. **Rollout**: probar
   advance + rollback.
3. **Desafio**: integrar
   con Argo Rollouts.

## Lecturas recomendadas

- "Canary Deployments" (AWS, 2024)
- "Argo Rollouts" (Argo, 2024)
- "Blue-Green Deploy" (Martin Fowler, 2024)

---

> 📚 **Adaptación al español de la lección [Shadow Canary Progressive]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).