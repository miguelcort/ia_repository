# Benchmarks SWE-bench GAIA

> SWE-bench: code agent benchmark (GitHub issues + pass@k + multi-language + Standard + Production). GAIA: general assistant benchmark (multi-level + reasoning + tool + Standard + Production). Niveles GAIA: Level 1 (simple +Easy), Level 2 (medium +Moderate), Level 3 (hard +Complex +Multi-step +External tool). pass@k: sample k candidates -> run each -> pass if any correct. +Standard, +Reproducible, +Comparable, +Reliable, +Standardized, +Production. Variants: SWE-bench, GAIA, WebArena (+web +navigation +realistic +Production), OSWorld (+OS +computer +realistic +Production), lm-eval, HELMET, OpenCompass, custom. Frameworks: swebench, gaia, webarena, osworld, lm-eval, helmet, opencompass, bigcode-evaluation-harness. +Production: standard 2024-25. +Use cases: agent, code, general, web, OS, evaluation, leaderboard. Decision: code -> SWE-bench, general -> GAIA, web -> WebArena, OS -> OSWorld, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + benchmarks.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar swebench_score (pass@k).
- Implementar gaia_score per level.
- Implementar success_rate overall.
- Diagnosticar SWE-bench vs GAIA vs WebArena vs OSWorld.

## Constrúyelo

```python
def swebench_score(predictions, ground_truth, k_values=(1, 5, 10)):
    correct = [i for i, (p, g) in enumerate(zip(predictions, ground_truth)) if p == g]
    return {f"pass@{k}": len(correct) / len(ground_truth) for k in k_values if k <= len(ground_truth)}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: benchmarks
fase: 14
leccion: 19
---

1. SWE-bench pass@k.
2. GAIA multi-level.
3. Success rate.
4. +Standard.
5. +Reproducible.
```

## Ejercicios

1. **SWE-bench**: probar
   SWE-bench con agent.
2. **GAIA**: evaluar
   agent en GAIA.
3. **Desafio**: full
   benchmark suite.

## Lecturas recomendadas

- "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (Jimenez et al., 2024)
- "GAIA: A Benchmark for General AI Assistants" (Mialon et al., 2024)
- "lm-evaluation-harness" (EleutherAI, 2024)
- "HELMET: Long-Context Evaluation" (Yale, 2024)

---

> 📚 **Adaptación al español de la lección [Benchmarks SWE-bench GAIA]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).