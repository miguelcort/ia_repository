# Kill switches & canaries

> Kill switches + canaries: (1) Feature flags (toggle+rollout %+allowlist), (2) Gradual rollout (10%+50%+100%), (3) Automatic rollback (error rate+latency), (4) Circuit breakers (open+cooldown), (5) Health checks (endpoint+threshold), (6) Canary analysis (variant+compare). FeatureFlag: name (string+unique), enabled (bool+toggle), rollout_pct (0-100+bucket), allowlist (set+VIP), is_enabled_for(user_id) check denylist+allowlist+enabled, bucket = hash(user_id) % 100, return bucket < rollout_pct, set_rollout(pct) 0-100, kill() disable+reset, enable() toggle. Canary analysis: canary_pct (0-100+route), error_threshold (0-1+compare), min_requests (int+min), record(req, variant, success) append+time, should_promote check min+err_rate+compare baseline, should_rollback check min+err_rate>threshold. Criterios: kill switch = emergency+instant+off, canary = gradual+compare+promote, health = continuous+fail+threshold. Decision: instant off -> kill switch, gradual -> canary, continuous -> health, mix -> todos, production -> todos. Frameworks: launchdarkly, unleash, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + safety.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar FeatureFlag con rollout % + allowlist + kill + enable.
- Implementar CanaryDeploy con route + record + promote/rollback.
- Implementar HealthCheck con threshold + recovery.
- Diagnosticar kill + canary + health.
- Diagnosticar criteria.

## Constrúyelo

```python
class FeatureFlag:
    def is_enabled_for(self, user_id):
        if user_id in self.allowlist:
            return True
        if not self.enabled:
            return False
        bucket = (hash(user_id) % 10000) / 100.0
        return bucket < self.rollout_pct
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: kill-switches-canaries
fase: 15
leccion: 14
---

1. FeatureFlag + rollout %.
2. CanaryDeploy + promote/rollback.
3. HealthCheck + threshold.
4. +Production.
```

## Ejercicios

1. **FeatureFlag**: probar
   rollout 0/50/100.
2. **Canary**: probar
   promote/rollback.
3. **Desafio**: integrar
   con LaunchDarkly o Unleash.

## Lecturas recomendadas

- "Feature Flag Best Practices" (LaunchDarkly, 2024)
- "Canary Deployments" (AWS, 2024)
- "Health Checks" (Google SRE Book, 2016)

---

> 📚 **Adaptación al español de la lección [Kill Switches & Canaries]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).