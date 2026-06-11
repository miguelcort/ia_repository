# Hybrid memory Mem0

> Mem0 (2024): hybrid memory layer para AI. Additive memory: (1) extract facts from conversation (LLM extraction), (2) add to memory store (+persistent), (3) retrieve on query (cosine sim top-k). Long-term + short-term (rolling window) tiers. Vector (embeddings) + graph (entities + relations). +Personalization, +Adaptive, +User-specific, +Scalable, +Reliable. +Hybrid, +Additive, +Persistent, +Long-term, +Modular. Variants: Mem0 seminal, MemGPT (Packer 2023 +hierarchical +OS-like +long context), Letta (2024 +blocks +modular +sleep-time), LangChain memory, custom. Frameworks: mem0, memgpt, letta, langchain, smolagents. +Production: standard 2024-25. +Use cases: agent, RAG, personalization, chat history, long-term, document. Decision: additive -> Mem0, long context -> MemGPT, modular -> Letta, production -> combinacion. Trade-offs: cada uno + specialty, custom + simple. 2025: +MCP + A2A + native + memory.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/07, 14/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar cosine_sim.
- Implementar Mem0Memory con add y search.
- Implementar short_term rolling.
- Implementar add_to_graph (entities + relations).
- Diagnosticar Mem0 vs MemGPT vs Letta.

## Constrúyelo

```python
class Mem0Memory:
    def _embed(self, content):
        rng = sum(ord(c) for c in str(content))
        return [((rng + i * 17) % 100) / 100.0 for i in range(self.embed_dim)]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mem0
fase: 14
leccion: 09
---

1. Hybrid memory.
2. Additive extraction.
3. Long + short term.
4. Vector + graph.
5. +Personalization.
```

## Ejercicios

1. **Mem0**: usar Mem0
   con personalized memory.
2. **Graph**: agregar
   knowledge graph.
3. **Desafio**: agent
   personalized con Mem0.

## Lecturas recomendadas

- "Mem0: Memory Layer for AI" (Mem0, 2024)
- "MemGPT: Towards LLMs as Operating Systems" (Packer, 2023)
- "Letta: Memory Blocks" (Letta, 2024)
- "LangChain Memory" (LangChain, 2024)

---

> 📚 **Adaptación al español de la lección [Hybrid Memory Mem0]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).