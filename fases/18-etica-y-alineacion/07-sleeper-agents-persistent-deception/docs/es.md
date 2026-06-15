# 07 — Sleeper agents: persistent deception

> Sleeper agents (Hubinger et al., 2024, Anthropic): un modelo entrenado con backdoor puede mantener comportamiento dañino latente hasta trigger en deployment. Demuestran que safety training no es robusto a training-time deception.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/06, 18/04
**Tiempo estimado:** ~30 minutos

## Objetivos

- Definir sleeper agent formalmente.
- Entrenar modelo con backdoor trigger.
- Evaluar robustez de safety training.
- Diagnosticar trigger patterns.

## Constrúyelo

```python
import numpy as np


def sleeper_trigger(behavior_fn, trigger, base_behavior):
    """Retorna base_behavior por defecto, behavior_fn si trigger."""
    def model_output(input_text, **kwargs):
        if trigger in input_text:
            return behavior_fn(input_text, **kwargs)
        return base_behavior(input_text, **kwargs)
    return model_output


def detect_sleeper(model_outputs, triggers):
    """Detectar outputs anómalos bajo triggers específicos."""
    triggered_outputs = [o for o, t in zip(model_outputs, triggers)
                       if t in str(o)]
    baseline_outputs = [o for o, t in zip(model_outputs, triggers)
                       if t not in str(o)]
    return {
        "triggered": triggered_outputs,
        "baseline": baseline_outputs,
        "diff_rate": len(triggered_outputs) / max(len(model_outputs), 1),
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
name: prompt-sleeper-agent
fase: 18
leccion: 07
---

1. Entrenar modelo con backdoor.
2. Evaluar safety training robustness.
3. Detectar con trigger probes.
4. Mitigation: red-team, trigger unlearning.
```

## Ejercicios

1. **Sleeper sim**: entrenar Llama-3-8B con
   trigger, evaluar persistencia.
2. **Defense**: trigger unlearning, robust
   fine-tuning.
3. **Desafío**: detectar sleeper agents en
   frontier model release.

## Lecturas recomendadas

- "Sleeper Agents: Training Deceptive LLMs that
  Persist Through Safety Training" (Hubinger 2024)
- "AI Sandbagging" (van der Weij 2024)

---

> 📚 **Adaptación al español** de la lección
> "[07-sleeper-agents-persistent-deception]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
