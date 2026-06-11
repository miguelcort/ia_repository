# Memory blocks sleep time compute

> Memory blocks (Letta 2024): named blocks con label (persona, human, facts, custom) + value (string editable) + limit (max chars, truncate) + updated_at. Sleep-time compute: periodic consolidation (consolidation_fn processes blocks) + background agent (search blocks for query) + +Offline optimization + -Latency impact + +Quality + +Reliable + +Optimized. +Modular, +Structured, +Per-block, +Standardized, +Stable, +Reliable. Variants: blocks, sleep-time, background agent, periodic, Letta, Mem0, custom. Frameworks: letta, mem0, langchain, smolagents. +Production: standard 2024-25. +Use cases: agent, RAG, long context, document, chat history. Decision: structured -> blocks, long context -> MemGPT tiers, simple -> custom, production -> blocks o MemGPT. Trade-offs: cada uno + specialty, blocks + structured, custom + simple. 2025: +MCP + A2A + native + memory.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/07
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MemoryBlock class.
- Implementar MemoryBlocksManager.
- Implementar sleep_time_consolidate.
- Implementar background_agent.
- Diagnosticar blocks vs MemGPT vs custom.

## Constrúyelo

```python
class MemoryBlock:
    def update(self, new_value):
        if len(new_value) > self.limit:
            new_value = new_value[:self.limit]
        self.value = new_value
        self.updated_at = time.time()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: memory-blocks
fase: 14
leccion: 08
---

1. Memory blocks.
2. Update truncate.
3. Sleep-time consolidate.
4. Background agent.
5. +Offline.
```

## Ejercicios

1. **Letta**: usar Letta
   memory blocks.
2. **Sleep-time**: implementar
   background agent.
3. **Desafio**: full
   long-running agent.

## Lecturas recomendadas

- "Letta: Open Source Memory Blocks" (Letta, 2024)
- "MemGPT Sleep-Time Compute" (Packer, 2023)
- "LangChain Memory Blocks" (LangChain, 2024)
- "Background Agents" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [Memory Blocks Sleep Time Compute]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).