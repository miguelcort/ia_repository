# Context engineering

> Context engineering (Anthropic 2024, Shopify): optimizar el context window completo (system + few-shot + tools + RAG + memory + history). Vs prompt engineering (solo el texto). Techniques: prioritization, sliding window, summarization, compression. Long context: 128K-1M (Llama 3.1, Gemini 1.5, Claude 3.5, Qwen 2.5), 2M (Gemini 1.5 Pro), 100M (Magic). Benchmarks: LongBench, RULER, SCROLLS, LV-Eval. Memory: mem0 (2024, scalable), Letta, Zep, LangGraph. Compresion: LLMLingua-2 (LLM-based, 4-20x), LongLLMLingua (long context, 30x), RECOMP (retrieval + compression). Frontier: 10M-100M context, +stateful, +agentic.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/04-embeddings
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar context budget check.
- Implementar context prioritization.
- Implementar sliding window context.
- Diagnosticar long context strategies.

## Constrúyelo

```python
def context_priority(items, max_tokens):
    sorted_items = sorted(items, key=lambda x: -x.get("priority", 0))
    used = 0
    out = []
    for item in sorted_items:
        cost = len(item["content"]) // 4
        if used + cost <= max_tokens:
            out.append(item)
            used += cost
    return out, used
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: context-engineering
fase: 11
leccion: 05
---

1. Context: system + few-shot + tools + RAG + memory.
2. Token budget, prioritization.
3. Sliding window, summarization.
4. Long context 128K-1M.
5. mem0, Letta memory.
```

## Ejercicios

1. **Context budget**: implementar
   priority-based context selection.
2. **Sliding window**: sliding
   context para long conv.
3. **Desafio**: long context
   QA con RAG.

## Lecturas recomendadas

- "Effective Long Context via Context Engineering" (Anthropic, 2024)
- "LLMLingua: Compressing Prompts for Accelerated Inference" (Microsoft, 2023)
- "MemGPT: Towards LLMs as Operating Systems" (Packer et al., 2023)
- "mem0: The Memory Layer for AI" (2024)

---

> 📚 **Adaptación al español** de la lección "[Context Engineering]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).