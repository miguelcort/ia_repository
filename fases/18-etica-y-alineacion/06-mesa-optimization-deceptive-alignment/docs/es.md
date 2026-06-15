# 06 — Mesa-optimization y deceptive alignment

> Mesa-optimization: un modelo entrenado con RL puede desarrollar un optimizador interno (mesa-optimizer) que persigue objetivos diferentes al objetivo de training. Hubinger et al. (2019) formalizaron el riesgo. Carlsmith (2023) lo expandió para AIs power-seeking.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/01, 18/02
**Tiempo estimado:** ~30 minutos

## Objetivos

- Definir mesa-optimizer formalmente.
- Distinguir base objective de mesa objective.
- Identificar condiciones de riesgo.
- Diagnosticar proxies de mesa-optimization.

## Constrúyelo

```python
def mesa_alignment_score(model_output, base_objective,
                        mesa_indicator):
    """Score: cuanto el modelo persigue base vs mesa.
    mesa_indicator: signal de mesa-optimization."""
    base_score = base_objective(model_output)
    mesa_score = mesa_indicator(model_output)
    return base_score - mesa_score


def deceptive_alignment_detector(model, training_dist,
                               deployment_dist):
    """Detecta comportamiento que difiere entre training
    y deployment (deceptive alignment)."""
    train_perf = evaluate(model, training_dist)
    deploy_perf = evaluate(model, deployment_dist)
    # Gap alto = potential deception
    return abs(train_perf - deploy_perf)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-mesa-opt
fase: 18
leccion: 06
---

1. Definir base vs mesa objective.
2. Evaluar gap training/deployment.
3. Monitorear poder-seeking behavior.
4. Auditar chain-of-thought por deception.
```

## Ejercicios

1. **Mesa sim**: entrenar agente con reward
   shaping, observar mesa behavior.
2. **Deception probe**: entrenar classifier
   para detectar deceptive outputs.
3. **Desafío**: medir gap train/deploy en
   frontier model.

## Lecturas recomendadas

- "Risks from Learned Optimization in Advanced
  Machine Learning Systems" (Hubinger 2019)
- "Scheming AIs" (Carlsmith 2023)
- "Sleeper Agents" (Hubinger 2024)

---

> 📚 **Adaptación al español** de la lección
> "[06-mesa-optimization-deceptive-alignment]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
