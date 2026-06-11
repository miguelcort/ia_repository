# Parallel and streaming tool calls

> Parallel + streaming tool calls: parallel (asyncio.gather / ThreadPoolExecutor, multiple tools 1 turn, -latency, -wall clock) + streaming (SSE chunks, token-by-token, assemble tool calls from deltas). OpenAI streaming (stream=True, tool_calls deltas, name concatenate + args char by char, id constant, index-based). Anthropic streaming (stream=True, tool_use blocks). +Insights: -Latency, +UX, +Throughput, +Efficiency, +Real-time, -Sequential, -Buffering. asyncio.gather: ejecuta multiple coroutines en paralelo, return list of results en mismo orden, -wall clock, +latency, +parallel, +reliability. Variants: asyncio.gather, asyncio.TaskGroup (3.11+), asyncio.wait, asyncio.as_completed. Frameworks: openai, anthropic, langchain, vLLM, transformers, asyncio, aiohttp, MCP. +Production: standard 2024-25. +Use cases: agent, RAG, automation, real-time, MCP. Trade-offs: parallel + efficiency, streaming + UX, sequential + simple. Hoy: SOTA 2024-25 standard. 2025: +MCP + A2A + native + real-time.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/02
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar parallel_execute con asyncio.gather.
- Implementar streaming_chunk (SSE format).
- Implementar assemble_tool_calls_from_deltas.
- Diagnosticar asyncio.gather variants.
- Diagnosticar streaming vs parallel.

## Constrúyelo

```python
async def parallel_execute(tool_calls, executor):
    tasks = [executor(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: parallel-streaming
fase: 13
leccion: 03
---

1. asyncio.gather.
2. SSE chunks.
3. Index-based assembly.
4. -Latency, +UX.
5. +Production.
```

## Ejercicios

1. **OpenAI streaming**:
   usar openai stream=True.
2. **Parallel**: probar
   asyncio.gather custom.
3. **Desafio**: parallel
   + streaming MCP.

## Lecturas recomendadas

- "OpenAI Streaming" (OpenAI Documentation, 2024)
- "Anthropic Streaming" (Anthropic Documentation, 2024)
- "Python asyncio" (Python Documentation)
- "asyncio.gather" (Python Documentation)

---

> 📚 **Adaptación al español de la lección [Parallel and Streaming Tool Calls]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).