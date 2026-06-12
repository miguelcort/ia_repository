# Group chat speaker selection

> Selection: (1) Round-robin (cycle+fair), (2) Random (stochastic+diverse), (3) LLM-driven (model decides+smart), (4) Moderator (central+controlled), (5) Weighted (probabilistic+custom), (6) Topic (expertise+match). round_robin: advance ((idx+n) % len)+wrap mod. moderator: if not last -> return mod, else -> advance. weighted: cumulative (cum += w) + random (r < cum). Criterios: Round-robin = fair+simple+equal turns, Random = diverse+stochastic+equal, Moderator = controlled+central+on-topic, LLM = smart+contextual+cost, Topic = expertise+match. Decision: fair -> round-robin, diverse -> random, control -> moderator, smart -> LLM, expert -> topic. Criterios topic vs LLM: Topic = deterministic+expertise+fast, LLM = smart+contextual+cost. Decision: static -> topic, dynamic -> LLM, mix -> topic+LLM. Frameworks: langchain, crewai, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + selection.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/09
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar round_robin con wrap.
- Implementar random_speaker + moderator_speaker.
- Implementar weighted_speaker con cumulative.
- Implementar topic_speaker con expertise.
- Diagnosticar criteria.

## Constrúyelo

```python
def round_robin(members, current_index, n=1):
    if not members:
        return None, None
    new_index = (current_index + n) % len(members)
    return new_index, members[new_index]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: group-chat-speaker-selection
fase: 16
leccion: 10
---

1. round_robin + moderator.
2. weighted + topic.
3. random_speaker.
4. +Production.
```

## Ejercicios

1. **round_robin**: probar
   wrap.
2. **moderator**: probar
   control.
3. **Desafio**: implementar
   LLM-driven.

## Lecturas recomendadas

- "AutoGen: Group Chat" (Microsoft, 2024)
- "CrewAI: Process" (CrewAI, 2024)
- "Anthropic: Multi-Agent" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Group Chat Speaker Selection]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).