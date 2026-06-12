# A/B testing LLM features

> A/B testing: (1) Control (baseline), (2) Variant (new), (3) Statistical (p-value+z-score), (4) Sample size (power+MDE), (5) Conversion (rate), (6) A/B/n (multi). ABTest: control_name+variants+results+record (variant, success)+conversion_rate (success/total)+lift (variant) ((v-c)/c)+z_score (two-proportion)+is_significant (abs >= 1.96). required_sample_size: baseline+mde+z_alpha=1.96+z_beta=0.84+p_bar=(p1+p2)/2+formula. Diferencias: A/B = 2 arms+simple+no correction, A/B/n = 3++complex+Bonferroni. Criterios: A/B = 2 arms+simple+fast, A/B/n = 3++complex+Bonferroni, Sequential = continuous+peek, Multi-armed = adaptive+bandits. Decision: 2 -> A/B, 3+ -> A/B/n, continuous -> sequential, adaptive -> bandits. Frameworks: statsig, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + ab.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/20
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ABTest con control + variants.
- Implementar record + conversion_rate + lift.
- Implementar z_score + is_significant.
- Implementar required_sample_size.
- Diagnosticar A/B vs A/B/n.

## Constrúyelo

```python
def z_score(self, variant):
    c = self.results[self.control_name]
    v = self.results[variant]
    p_c = c["success"] / c["total"]
    p_v = v["success"] / v["total"]
    p_pool = (c["success"] + v["success"]) / (c["total"] + v["total"])
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / c["total"] + 1 / v["total"]))
    return (p_v - p_c) / se
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: ab-testing-llm
fase: 17
leccion: 21
---

1. ABTest.
2. record + lift.
3. z_score.
4. required_sample_size.
5. +Production.
```

## Ejercicios

1. **ABTest**: probar
   record + lift.
2. **Z-score**: probar
   significance.
3. **Desafio**: integrar
   con Statsig.

## Lecturas recomendadas

- "A/B Testing" (Kohavi, 2020)
- "Statsig" (Statsig, 2024)
- "A/B/n Tests" (Microsoft, 2024)

---

> 📚 **Adaptación al español de la lección [A/B Testing LLM Features]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).