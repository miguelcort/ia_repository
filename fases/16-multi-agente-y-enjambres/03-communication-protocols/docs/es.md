# Communication protocols

> Protocols: (1) Direct (1-to-1+mailbox), (2) Broadcast (1-to-N+inbox), (3) Pub/Sub (channels+subscribe), (4) Blackboard (shared state+key-value), (5) Request/Reply (sync+pattern), (6) Async (future+promise). Channel: name+subscribers+messages+subscribe(agent)+unsubscribe(agent)+publish(sender, content). PubSub: channels dict+create_channel(name) lazy+get_channel(name). Blackboard: entries dict+write(key, value, author)+read(key)+keys(). DirectMailbox: agent_id+inbox+receive(msg) filter+receiver match+broadcast. Criterios: Direct = 1-1+simple+sync, Pub/Sub = 1-N+dynamic+subscribe, Blackboard = N-N+shared state+key-value. Decision: 1-1 -> direct, 1-N -> pub/sub, N-N -> blackboard, mix -> pub/sub+blackboard. Frameworks: langchain, anthropic, openai, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + protocols.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/02
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Message con sender + receiver + content + id.
- Implementar Channel + PubSub con subscribe + publish.
- Implementar Blackboard con write + read.
- Implementar DirectMailbox con receive.
- Diagnosticar protocols.

## Constrúyelo

```python
class Channel:
    def subscribe(self, agent):
        if agent not in self.subscribers:
            self.subscribers.append(agent)

    def publish(self, sender, content):
        msg = Message(sender, "broadcast", content, channel=self.name)
        self.messages.append(msg)
        for sub in self.subscribers:
            sub.inbox.append(msg)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: communication-protocols
fase: 16
leccion: 03
---

1. Direct + Mailbox.
2. Pub/Sub + Channel.
3. Blackboard.
4. +Production.
```

## Ejercicios

1. **Channel**: probar
   subscribe + publish.
2. **Blackboard**: probar
   write + read.
3. **Desafio**: implementar
   request/reply con timeout.

## Lecturas recomendadas

- "Distributed Systems" (Tanenbaum, 2007)
- "ZeroMQ" (Hintjens, 2013)
- "Pub/Sub Patterns" (Microsoft, 2024)

---

> 📚 **Adaptación al español de la lección [Communication Protocols]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).