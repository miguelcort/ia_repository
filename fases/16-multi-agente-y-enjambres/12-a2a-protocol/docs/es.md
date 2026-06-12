# A2A protocol

> A2A: (1) Agent-to-Agent (Google+2025), (2) Agent cards (metadata+skills), (3) JSON-RPC (transport+standard), (4) Interop (cross-vendor+standard), (5) Modern ACL alt (replaces FIPA+modern). AgentCard: name+description+url+version+skills list+capabilities (streaming+pushNotifications)+add_skill(name, desc, examples). A2A message: jsonrpc 2.0+id (UUID)+method message/send+params with message (from+to+parts+role)+text_part+data_part+parse_a2a_message. Ventajas A2A vs FIPA: JSON-RPC (HTTP+standard), agent cards (discovery+metadata), modern (2025+active), web-friendly (REST+HTTPS), interop (multi-vendor+standard). Criterios: A2A = modern+interop+web, FIPA ACL = standards+formal+IEEE, MCP = tools+Anthropic+modern, Custom = specific+internal. Decision: modern -> A2A, formal -> FIPA, tools -> MCP, specific -> custom, mix -> A2A+MCP. Frameworks: google, langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + A2A.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/11
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar create_agent_card con skills + capabilities.
- Implementar add_skill.
- Implementar a2a_message con JSON-RPC.
- Implementar text_part + data_part.
- Implementar parse_a2a_message.
- Diagnosticar A2A vs FIPA.

## Constrúyelo

```python
def a2a_message(sender, receiver, parts, message_id=None, role="agent"):
    return {
        "jsonrpc": "2.0",
        "id": message_id or str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": role,
                "parts": parts,
                "from": sender,
                "to": receiver,
            }
        },
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: a2a-protocol
fase: 16
leccion: 12
---

1. AgentCard.
2. A2A JSON-RPC.
3. text + data parts.
4. +Production.
```

## Ejercicios

1. **Card**: probar
   skills + add.
2. **Message**: probar
   parse.
3. **Desafio**: integrar
   con Google A2A server.

## Lecturas recomendadas

- "A2A Protocol" (Google, 2025)
- "JSON-RPC 2.0" (JSON-RPC, 2009)
- "Agent Cards" (Google, 2025)

---

> 📚 **Adaptación al español de la lección [A2A Protocol]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).