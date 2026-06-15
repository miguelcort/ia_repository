# 10 — Multi-agent software team

> Multi-agent software team: PM, architect, developer, QA, code reviewer coordinan via shared blackboard. Frameworks: MetaGPT, ChatDev, AutoGen, CrewAI, LangGraph. Simula SDLC completo. Resuelve tareas como "build a Flask app with auth".

**Tipo:** Capstone
**Lenguajes:** Python
**Prerrequisitos:** Fase 16 (multi-agent), Fase 14, Fase 11
**Tiempo estimado:** 30 horas

## Objetivos

- Diseñar roles y comunicación.
- Blackboard o message-passing.
- Coordination y conflict resolution.
- Eval sobre SoftwareDev benchmark.

## El problema

Multi-agent teams (MetaGPT, ChatDev, AutoGen) simulan
SDLC: ProductManager (requirements), Architect (design),
Developer (code), QA (tests), Reviewer (PR). Patrones:
(1) Blackboard: shared state. (2) Message-passing:
pub-sub. (3) Sequential pipeline. (4) Hierarchical
(manager + workers). (5) Debate (multi-proposer,
critic). Frameworks: LangGraph (graph state),
CrewAI (roles), MetaGPT (SOPs), AutoGen (conversable).

## Constrúyelo

```python
from crewai import Agent, Crew, Task


def build_software_team():
    pm = Agent(role="ProductManager", goal="Spec the app",
              backstory="Experienced PM")
    arch = Agent(role="Architect", goal="Design system",
                backstory="Senior architect")
    dev = Agent(role="Developer", goal="Write code",
               backstory="Full-stack dev")
    qa = Agent(role="QA", goal="Test thoroughly",
              backstory="QA engineer")
    return Crew(agents=[pm, arch, dev, qa],
               tasks=[Task("Write spec"),
                     Task("Design"),
                     Task("Implement"),
                     Task("Test")])
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-software-team
fase: 19
leccion: 10
---

1. Roles (PM, arch, dev, QA).
2. Blackboard coordination.
3. SOPs o task graph.
4. Eval métricas (pass rate, time).
```

## Ejercicios

1. **CrewAI**: app Flask con auth.
2. **LangGraph**: custom
   coordination.
3. **Desafío**: 5-agent team en
   multi-file project.

## Lecturas recomendadas

- "MetaGPT" (Hong 2023)
- "ChatDev" (Qian 2023)
- "AutoGen" (Wu 2023, Microsoft)
- "CrewAI" (2024)

---

> 📚 **Adaptación al español** de la lección
> "[10-multi-agent-software-team]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
