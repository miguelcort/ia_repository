# METR external evaluation

> METR: (1) Independent (third party+external), (2) Task suite (HCAST+RE-Bench+SWE-bench+GAIA), (3) Time horizon (calibrated+median), (4) Agent benchmarks (coding+research+general), (5) HCAST + RE-Bench (METR+independent). Benchmarks: HCAST (human-calibrated+agent solver+time horizon metric), RE-Bench (research engineering+ML+score metric), SWE-bench (software eng+real GitHub issues+resolve rate), GAIA (general AI assistants+multi-modal+accuracy). time_horizon: ratio duration/max(model_time, 0.01)+sort+median (even -> avg mid, odd -> mid)+edge cases (empty -> 0, partial -> available only). Criterios: METR = independent+third-party+audit, Vendor = marketing+self-report+optimistic, Internal = specific+custom+in-house. Decision: audit -> METR, marketing -> vendor, specific -> internal, mix -> all. Frameworks: metr, anthropic, openai, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + evaluation.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15/20
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar METR_BENCHMARKS con 4 benchmarks.
- Implementar time_horizon_score con median.
- Implementar evaluate_model.
- Diagnosticar benchmarks.
- Diagnosticar METR vs vendor vs internal.

## Constrúyelo

```python
def time_horizon_score(tasks, model_times):
    horizons = []
    for task, duration in tasks.items():
        if task in model_times:
            horizon = duration / max(model_times[task], 0.01)
            horizons.append(horizon)
    if not horizons:
        return 0.0
    horizons.sort()
    n = len(horizons)
    if n % 2 == 0:
        return (horizons[n//2 - 1] + horizons[n//2]) / 2
    return horizons[n//2]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: metr-external-evaluation
fase: 15
leccion: 21
---

1. HCAST + RE-Bench + SWE
   + GAIA.
2. time_horizon.
3. evaluate_model.
4. +Production.
```

## Ejercicios

1. **METR benchmarks**: probar
   los 4 benchmarks.
2. **time_horizon**: probar
   median.
3. **Desafio**: integrar
   con tu eval pipeline.

## Lecturas recomendadas

- "METR: HCAST" (METR, 2024)
- "METR: RE-Bench" (METR, 2024)
- "Time Horizon Evaluations" (METR, 2024)

---

> 📚 **Adaptación al español de la lección [METR External Evaluation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).