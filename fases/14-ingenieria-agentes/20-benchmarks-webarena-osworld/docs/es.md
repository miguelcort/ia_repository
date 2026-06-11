# Benchmarks WebArena OSWorld

> WebArena (CMU 2023): web benchmark realistic (sites: shopping, reddit, gitlab, maps + navigation + multi-step + success per site + comparable). OSWorld (2024): OS benchmark realistic (apps: chrome + vscode + gimp + terminal + vlc + computer use + success per app + production). task_completion_rate: per task con partial credit (0.5 for partial matches). +Realistic, +Standard, +Production, +Reliable, +Standardized, +Comparable. Variants: WebArena, VisualWebArena, Mind2Web, OSWorld, AndroidWorld, SWE-bench, GAIA, lm-eval, HELMET, custom. Frameworks: webarena, osworld, lm-eval, swebench, gaia, helmet. +Production: standard 2024-25. +Use cases: agent, web, OS, code, general, computer, evaluation. Decision: web -> WebArena, OS -> OSWorld, code -> SWE-bench, general -> GAIA, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + benchmarks.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/19
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar webarena_score (per site).
- Implementar osworld_score (per app).
- Implementar task_completion_rate con partial credit.
- Diagnosticar WebArena vs OSWorld vs SWE-bench vs GAIA.

## Constrúyelo

```python
def task_completion_rate(predictions, ground_truth, partial_credit=True):
    total = 0
    for p, g in zip(predictions, ground_truth):
        if p == g.get("answer"):
            total += 1.0
        elif partial_credit and g.get("partial") and p:
            total += 0.5
    return total / len(ground_truth)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: webarena-osworld
fase: 14
leccion: 20
---

1. WebArena sites.
2. OSWorld apps.
3. Success per site/app.
4. Partial credit.
5. +Realistic.
```

## Ejercicios

1. **WebArena**: probar
   WebArena con agent.
2. **OSWorld**: evaluar
   agent en OSWorld.
3. **Desafio**: full
   benchmark suite.

## Lecturas recomendadas

- "WebArena: A Realistic Web Environment for Building Autonomous Agents" (Zhou et al., CMU, 2023)
- "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments" (Xie et al., 2024)
- "VisualWebArena" (Koh et al., 2024)
- "Mind2Web: Towards a Generative Agent Factory" (Deng et al., 2023)

---

> 📚 **Adaptación al español de la lección [Benchmarks WebArena OSWorld]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).