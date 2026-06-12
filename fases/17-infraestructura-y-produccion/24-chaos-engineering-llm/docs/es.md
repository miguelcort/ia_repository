# Chaos engineering LLM

> Chaos LLM: (1) Inject (failures), (2) Latency (slow), (3) Malformed (bad resp), (4) Recovery (resilient), (5) Game days (drills). ChaosExperiment: name+hypothesis+injectors list+results list+add_injector+run (target_fn, n_trials) iterate+try/except+success rate. injectors: inject_error (raise)+inject_latency (sleep)+inject_malformed (return bad). ChaosMonkey: experiments+register+run_all. Ventajas chaos vs testing: Real failures (production+unexpected), Recovery (validate+drills), Blast radius (limit), Confidence (team+system), Game days (practice). Criterios: Chaos = production+critical+risk-aware, No chaos = dev+cheap+small. Decision: prod -> chaos, dev -> no, mix -> chaos+tests. Frameworks: chaos-toolkit, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + chaos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/23
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ChaosExperiment con name + hypothesis + injectors.
- Implementar injectors (error, latency, malformed).
- Implementar ChaosMonkey con register + run_all.
- Diagnosticar advantages.
- Diagnosticar criteria.

## Constrúyelo

```python
def run(self, target_fn, n_trials=10):
    successes = 0
    for _ in range(n_trials):
        try:
            for inj in self.injectors:
                inj()
            target_fn()
            successes += 1
        except Exception as e:
            self.results.append({"ok": False, "error": str(e)})
    return successes / n_trials
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: chaos-engineering-llm
fase: 17
leccion: 24
---

1. ChaosExperiment.
2. injectors.
3. ChaosMonkey.
4. +Production.
```

## Ejercicios

1. **Experiment**: probar
   run.
2. **Injector**: probar
   error.
3. **Desafio**: integrar
   con chaos-toolkit.

## Lecturas recomendadas

- "Chaos Engineering" (Basiri, 2016)
- "Chaos Toolkit" (Chaos Toolkit, 2024)
- "LLM Resilience" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Chaos Engineering LLM]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).