# Long horizon agents

> Long-horizon agents: tareas que duran horas/dias. Components: (1) persistence (state survives restarts), (2) checkpointing (save state periodically), (3) recovery (restore from checkpoint on failure), (4) human-in-the-loop (interrupt + approve), (5) +production. Variants: (1) Voyager (Wang 2023 +curriculum +code +Minecraft), (2) Auto-GPT (Significant Gravitas 2023 +autonomous +tools +internet), (3) Adept (+actions +UI +workflow), (4) Devin (Cognition 2024 +software engineer +long-horizon +coding), (5) smolagents, (6) langgraph, (7) custom. +Long, +Persistent, +Reliable, +Autonomous, +Scalable, +Production. Frameworks: voyager, auto-gpt, adept, devin, smolagents, langgraph. +Production: standard 2024-25. +Use cases: research, coding, automation, long-running. Decision: curriculum -> Voyager, autonomous -> Auto-GPT, actions -> Adept, coding -> Devin, production -> combinacion. Trade-offs: cada uno + specialty. 2025: +MCP + A2A + native + long-horizon.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar LongHorizonTask con step/checkpoint/restore.
- Diagnosticar Voyager vs Auto-GPT vs Adept vs Devin.
- Diagnosticar checkpointing + recovery.

## Constrúyelo

```python
class LongHorizonTask:
    def checkpoint(self):
        cp = {
            "name": self.name,
            "completed": self.completed,
            "state": dict(self.state),
            "timestamp": time.time(),
        }
        self.checkpoints.append(cp)
        return cp
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: long-horizon
fase: 15
leccion: 01
---

1. Long-horizon.
2. Checkpoint + restore.
3. Voyager, Auto-GPT, Devin.
4. HITL.
5. +Production.
```

## Ejercicios

1. **Long-horizon**: implementar
   task con checkpointing.
2. **Recovery**: probar
   restore desde checkpoint.
3. **Desafio**: full
   long-horizon agent.

## Lecturas recomendadas

- "Voyager: An Open-Ended Embodied Agent" (Wang et al., 2023)
- "Auto-GPT: Autonomous GPT-4" (Significant Gravitas, 2023)
- "Devin: AI Software Engineer" (Cognition, 2024)
- "Adept: Actions for LLMs" (Adept, 2024)

---

> 📚 **Adaptación al español de la lección [Long Horizon Agents]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).