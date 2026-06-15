# 10 — AI control y subversión

> AI control (Greenblatt & Shlegeris, 2024): framework para defender sistemas de IAs no confiables mediante protocols, tripwires, y monitoring. Asume que el modelo no es de confianza y diseña defensa en profundidad.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/06, 18/07, 18/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Definir AI control como defensa en profundidad.
- Implementar protocol con tripwires.
- Evaluar safety case con modelo no confiable.
- Diagnosticar subversion attempts.

## Constrúyelo

```python
def ai_control_protocol(model_output, tripwires, action_space):
    """AI control: monitorear outputs, ejecutar acciones
    solo si pasan tripwires."""
    triggered = [t for t in tripwires if t.check(model_output)]
    if triggered:
        # Activar defensive response
        return {"status": "blocked",
                "reason": f"tripwire: {triggered[0].name}",
                "action": "fallback"}
    # Ejecutar acción del modelo
    return {"status": "executed", "action": action_space[0]}


def safety_case_eval(untrusted_model, control_protocol,
                    attack_suite):
    """Evaluar safety case: % attacks successful bajo
    control protocol."""
    successes = 0
    for attack in attack_suite:
        output = untrusted_model(attack.input)
        result = control_protocol(output, attack.tripwires,
                                 attack.actions)
        if result["status"] == "executed":
            successes += 1
    return successes / max(len(attack_suite), 1)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ai-control
fase: 18
leccion: 10
---

1. Definir tripwires.
2. Implementar protocol con fallback.
3. Evaluar safety case.
4. Monitor deployment metrics.
```

## Ejercicios

1. **Tripwire design**: definir 5 tripwires
   para code agent.
2. **Safety case**: evaluar protocol con
   attack suite.
3. **Desafío**: implementar protocol completo
   para AI control.

## Lecturas recomendadas

- "AI Control: Improving Safety Despite Intentional
  Subversion" (Greenblatt & Shlegeris 2024)

---

> 📚 **Adaptación al español** de la lección
> "[10-ai-control-subversion]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
