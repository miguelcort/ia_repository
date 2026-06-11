# MCP sampling

> MCP sampling: server asks client to run LLM (recursive LLM use) via sampling/createMessage method. Request: messages, modelPreferences (costPriority/speedPriority/intelligencePriority 0-1, hints optional), maxTokens, systemPrompt, includeContext (none/thisServer/allServers). Response: model, content, stopReason (endTurn/maxTokens), role. Human-in-the-loop: client can prompt user for approval before executing. +Recursive, +Flexible, +HITL, +Cost-aware, +Cheap, +Fast, +Capable, +Optimized, +Tunable. Frameworks: mcp, fastmcp, anthropic, openai, langchain. +Production: standard 2024-25. +Use cases: agent, RAG, automation, cost optimization. Decision: recursive -> sampling, simple -> direct, HITL -> sampling, production -> ambos. Trade-offs: sampling + flexible, direct + simple. Hoy: SOTA 2024-25 standard. 2025: +MCP + A2A + native.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 13/07, 13/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar make_sampling_request con model preferences.
- Implementar make_sampling_response.
- Implementar model_preferences (cost/speed/intelligence).
- Implementar include_context_options.
- Implementar sampling_with_human_approval.
- Diagnosticar sampling vs direct call.

## Constrúyelo

```python
def make_sampling_request(messages, model_prefs=None, max_tokens=1024,
                          system_prompt=None, temperature=0.7, include_context="none"):
    params = {
        "messages": messages,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "includeContext": include_context,
    }
    if model_prefs:
        params["modelPreferences"] = model_prefs
    if system_prompt:
        params["systemPrompt"] = system_prompt
    return {"jsonrpc": "2.0", "method": "sampling/createMessage", "params": params, "id": 1}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mcp-sampling
fase: 13
leccion: 11
---

1. Server asks client.
2. sampling/createMessage.
3. Recursive LLM use.
4. HITL via client.
5. Model preferences.
```

## Ejercicios

1. **Sampling**: implementar
   recursive sampling.
2. **HITL**: agregar approval.
3. **Desafio**: cost-aware
   routing.

## Lecturas recomendadas

- "MCP Sampling Specification" (Anthropic, 2024)
- "Recursive LLM Calls" (Anthropic, 2024)
- "Human-in-the-Loop" (Anthropic, 2024)
- "Model Preferences" (Anthropic, 2024)

---

> 📚 **Adaptación al español de la lección [MCP Sampling]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).