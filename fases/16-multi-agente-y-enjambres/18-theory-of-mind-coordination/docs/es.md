# Theory of mind coordination

> ToM: (1) Beliefs (model+what they think), (2) Desires (goals+wants), (3) Intentions (plan+action), (4) Recursive (nested+what they think I think), (5) Common ground (shared+mutual). MindModel: beliefs Dict+desires Set+intentions List+update_belief(key, value)+add_desire(desire)+set_intention(intention)+predict_action (last or wait). TheoryOfMind: self_id+self_mind+others_minds+model_other(other_id, initial)+update_other_belief+predict_other_action+recursive_depth+find_common_ground (intersect). Ventajas ToM vs reactive: predict (future action+anticipate), coordinate (plan together+shared), negotiate (understand+compromise), empathize (perspective+social), plan (long horizon+multi-step). Criterios: ToM = complex+social+predict, Messaging = simple+reactive+fast, Shared state = coordination+sync+audit. Decision: social -> ToM, simple -> messaging, coordination -> shared, mix -> ToM+shared. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + ToM.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/17
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MindModel con beliefs + desires + intentions.
- Implementar TheoryOfMind con self + others_minds.
- Implementar model_other + update + predict.
- Implementar recursive_depth + find_common_ground.
- Diagnosticar ToM vs reactive.
- Diagnosticar criteria.

## Constrúyelo

```python
class MindModel:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.beliefs = {}
        self.desires = set()
        self.intentions = []
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: theory-of-mind
fase: 16
leccion: 18
---

1. MindModel + ToM.
2. recursive_depth.
3. common_ground.
4. +Production.
```

## Ejercicios

1. **MindModel**: probar
   beliefs + predict.
2. **ToM**: probar
   model + predict.
3. **Desafio**: implementar
   recursive ToM depth 3.

## Lecturas recomendadas

- "Theory of Mind" (Premack, 1978)
- "Recursive ToM" (Gmytrasiewicz, 2002)
- "Social Reasoning" (Baron-Cohen, 1995)

---

> 📚 **Adaptación al español de la lección [Theory of Mind Coordination]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).