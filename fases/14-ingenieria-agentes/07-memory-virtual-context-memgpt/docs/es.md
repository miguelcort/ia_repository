# Memory virtual context MemGPT

> MemGPT (Packer 2023, virtual context management): hierarchical memory con OS-like page in/out. Tiers: (1) core (in-context limited 10-100, +fast access), (2) archival (out-of-context unlimited, +search via query), (3) recall (rolling window, +recent, -permanent). Operations: page out (core full -> oldest to archival), page in (query archival -> matching to core), search_archival (sin page in, -disruption). +Infinite context, +Hierarchical, +OS-like, +Production, +Efficient. Variants: MemGPT seminal, Letta (2024 MemGPT fork +open source +production), Mem0 (2024 +hybrid +additive +scalable), LangChain memory, custom. Frameworks: memgpt, letta, mem0, langchain, smolagents. +Production: standard 2024-25. +Use cases: agent, RAG, long context, document, chat history. Decision: hierarchical -> MemGPT/Letta, open -> Letta, scalable -> Mem0, production -> Mem0 o Letta. Trade-offs: cada uno + specialty, MemGPT + hierarchical, custom + simple. 2025: +MCP + A2A + native + memory.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar VirtualContextMemory class.
- Implementar add_to_core con page out automatico.
- Implementar page_in_from_archival y search_archival.
- Implementar add_to_recall rolling window.
- Diagnosticar MemGPT vs Letta vs Mem0.

## Constrúyelo

```python
def add_to_core(self, item):
    if len(self.core) >= self.core_size:
        oldest = self.core.pop(0)
        self.archival.append(oldest)
        self.page_out_count += 1
    self.core.append(item)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: memgpt
fase: 14
leccion: 07
---

1. Virtual context.
2. Hierarchical.
3. Page in/out.
4. Core + archival + recall.
5. +Infinite.
```

## Ejercicios

1. **MemGPT**: implementar
   MemGPT con Letta.
2. **Mem0**: probar Mem0
   hybrid memory.
3. **Desafio**: agent
   long-running con memory.

## Lecturas recomendadas

- "MemGPT: Towards LLMs as Operating Systems" (Packer et al., 2023)
- "Letta: Open Source Memory" (Letta, 2024)
- "Mem0: Memory Layer for AI" (Mem0, 2024)
- "LangChain Memory" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [Memory Virtual Context MemGPT]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).