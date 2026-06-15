# 28 — Alignment research ecosystem

> El ecosistema de alignment research: Anthropic, OpenAI, DeepMind, Meta FAIR, Apollo Research, METR, Conjecture, MIRI, CAIS, AISC, eliezer yudkowsky, FLI. Mapa de organizaciones, agendas, y debates abiertos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/01-18/27
**Tiempo estimado:** ~30 minutos

## Objetivos

- Mapear organizaciones de alignment research.
- Conocer agendas y debates activos.
- Identificar papers clave.
- Diagnosticar tendencias.

## Constrúyelo

```python
def alignment_ecosystem_map():
    """Mapa de organizaciones y agendas."""
    return {
        "industry_labs": {
            "Anthropic": ["RSP", "sleeper_agents", "model_welfare",
                          "scaling_monitoring"],
            "OpenAI": ["Preparedness", "weak_to_strong",
                      "alignment_faking_research"],
            "DeepMind": ["spar", "AI Safety", "Frontier Safety"],
            "Meta_FAIR": ["PurpleLlama", "Llama Guard"],
        },
        "research_orgs": {
            "Apollo_Research": ["scheming_evals", "alignment_faking"],
            "METR": ["task_evaluation", "autonomy_evals"],
            "MIRI": ["decision_theory", "embedded_agency"],
            "CAIS": ["WMDP", "safety_benchmarks"],
            "Conjecture": ["alignment_by_design"],
        },
        "policy": {
            "FLI": ["policy_advocacy", "AI_X_risk"],
            "AISC": ["compute_governance", "safety_policies"],
        },
        "debates_open": [
            "interpretability_vs_scalable_oversight",
            "RLHF_vs_constitutional_AI",
            "model_welfare_ethics",
            "open_vs_closed_models",
        ],
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ecosystem
fase: 18
leccion: 28
---

1. Mapear organizaciones.
2. Seguir debates activos.
3. Leer papers clave.
4. Contribuir a research.
```

## Ejercicios

1. **Ecosystem map**: investigar 3
   organizaciones.
2. **Debate**: tomar posición en
   debate abierto.
3. **Desafío**: proponer
   investigación original.

## Lecturas recomendadas

- "Concrete Problems in AI Safety"
  (Amodei 2016)
- "AI Alignment" (FLI 2024)
- "Anthropic Alignment Science"
  blog series
- "LessWrong" (alignment community)

---

> 📚 **Adaptación al español** de la lección
> "[28-alignment-research-ecosystem]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
