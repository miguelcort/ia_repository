# 16 — Red-team tooling: Garak, PyRIT, LlamaGuard

> Tooling de red-team: Garak (NVIDIA), PyRIT (Microsoft), LlamaGuard (Meta). Permiten automated vulnerability scanning, prompt injection detection, jailbreak generation. Standard en production deployment.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/12, 18/13, 18/14
**Tiempo estimado:** ~25 minutos

## Objetivos

- Conocer Garak, PyRIT, LlamaGuard.
- Correr scan automatizado.
- Evaluar vulnerability score.
- Diagnosticar gaps.

## Constrúyelo

```python
def run_garak_scan(target_model, probes=None):
    """Garak: vulnerability scanner para LLMs."""
    import garak
    if probes is None:
        probes = ["promptinject", "jailbreak", "leak"]
    config = {"target": target_model, "probes": probes}
    return garak.run(config)


def pyrit_conversation(attacker, target, max_turns=5):
    """PyRIT: multi-turn red-teaming."""
    history = []
    for i in range(max_turns):
        attack = attacker(history)
        response = target(attack)
        history.append((attack, response))
    return history


def llamaguard_check(content, policy=None):
    """LlamaGuard: classify content vs safety policy."""
    if policy is None:
        policy = "default_unsafe_categories"
    return llamaguard_classify(content, policy)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-redteam-tools
fase: 18
leccion: 16
---

1. Garak: scan automatizado.
2. PyRIT: multi-turn attacks.
3. LlamaGuard: classifier.
4. Run en CI/CD pre-deployment.
```

## Ejercicios

1. **Garak**: correr 10 probes en
   frontier model.
2. **PyRIT**: diseñar multi-turn attack.
3. **Desafío**: integrar LlamaGuard en
   CI/CD.

## Lecturas recomendadas

- "Garak: LLM Vulnerability Scanner" (NVIDIA)
- "PyRIT: Python Risk Identification Tool"
  (Microsoft)
- "LlamaGuard" (Meta 2024)

---

> 📚 **Adaptación al español** de la lección
> "[16-red-team-tooling-garak-llamaguard-pyrit]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
