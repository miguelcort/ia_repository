# Evaluation coordination benchmarks

> Benchmarks: (1) MGSM (multi-step math+accuracy), (2) HumanEval-Multi (multi-file+pass@k), (3) MLE-bench (ML eng+score), (4) ChatBot Arena MT (multi-turn+elo), (5) SWE-bench Multi (multi-repo+resolve). normalize_score: metric-based (accuracy/pass@k/score/resolve_rate vs elo), accuracy=value/scale, elo=(value-800)/800, min/max 0-1. average_coordination_score: skip None+mean. Metricas: accuracy (MGSM), pass@k (HumanEval+code), score (MLE+ML), elo (ChatBot+chat), resolve_rate (SWE+repo). Criterios: Multi-agent = coordination+multi-file+multi-turn+multi-repo, Single = tasks+simple+cheap. Decision: coord -> multi, tasks -> single, mix -> both. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + eval.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/23
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar COORDINATION_BENCHMARKS con 5 benchmarks.
- Implementar normalize_score con metric-based scaling.
- Implementar average_coordination_score con skip None.
- Diagnosticar metricas.
- Diagnosticar multi-agent vs single.

## Constrúyelo

```python
def normalize_score(value, benchmark_key, scale=100):
    bench = COORDINATION_BENCHMARKS.get(benchmark_key)
    if not bench:
        return None
    metric = bench["metric"]
    if metric in ("accuracy", "pass@k", "score", "resolve_rate"):
        return min(1.0, max(0.0, value / scale))
    if metric == "elo":
        return min(1.0, max(0.0, (value - 800) / (1600 - 800)))
    return None
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: evaluation-coordination
fase: 16
leccion: 24
---

1. 5 benchmarks.
2. normalize_score.
3. average + skip None.
4. +Production.
```

## Ejercicios

1. **Benchmarks**: probar
   los 5.
2. **normalize**: probar
   elo.
3. **Desafio**: integrar
   con tu eval pipeline.

## Lecturas recomendadas

- "MGSM" (Cobbe, 2021)
- "HumanEval" (Chen, 2021)
- "MLE-bench" (OpenAI, 2024)

---

> 📚 **Adaptación al español de la lección [Evaluation Coordination Benchmarks]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).