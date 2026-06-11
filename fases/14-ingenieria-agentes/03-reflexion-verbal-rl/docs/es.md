# Reflexion verbal RL

> Reflexion (Shinn 2023): agents learn via verbal self-reflection. Memory of reflections (max size, FIFO eviction, format for prompt). Loop: trial -> reflection -> next trial. +Self-improving, +Adaptive, +Verbal RL (no gradient updates, -compute, +effective). Reflection types: (1) success ('I succeeded via X' +strategy), (2) failure ('I failed because Y' +reason), (3) mistake ('I made error Z' +diagnosis), (4) strategy ('New approach: A' +plan). +Diverse, +Quality. Variants: Reflexion, Self-Refine (Madaan 2023 +iterative +single-trial +quality), CRITIC (Gou 2024 +critic +tool-use +verification), Constitutional AI, RLAIF. Frameworks: langchain, openai, anthropic, autogen, crewai. +Production: standard 2024-25. +Use cases: agent, RAG, coding, decision making. Decision: memory -> Reflexion, quality -> Self-Refine, verification -> CRITIC, production -> Reflexion o Self-Refine. Trade-offs: cada uno + specialty, Reflexion + memory, single-shot + simple. 2025: +MCP + A2A + native + variants.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ReflexionMemory con max_size.
- Implementar generate_reflection (success/failure/mistake).
- Implementar reflexion_loop (trial -> reflection -> next).
- Diagnosticar Reflexion vs Self-Refine vs CRITIC.

## Constrúyelo

```python
class ReflexionMemory:
    def add(self, reflection):
        self.reflections.append({"timestamp": time.time(), "content": reflection})
        if len(self.reflections) > self.max_size:
            self.reflections = self.reflections[-self.max_size:]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: reflexion
fase: 14
leccion: 03
---

1. Self-reflection.
2. Memory of reflections.
3. Trial -> reflection ->
   next trial.
4. +Self-improving.
5. Verbal RL.
```

## Ejercicios

1. **Reflexion**: implementar
   Reflexion con LLM real.
2. **Self-Refine**: probar
   Self-Refine pattern.
3. **Desafio**: coding
   agent con Reflexion.

## Lecturas recomendadas

- "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023)
- "Self-Refine: Iterative Refinement with Self-Feedback" (Madaan et al., 2023)
- "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing" (Gou et al., 2024)
- "Constitutional AI" (Anthropic, 2022)

---

> 📚 **Adaptación al español de la lección [Reflexion Verbal RL]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).