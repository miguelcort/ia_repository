# FIPA ACL heritage

> FIPA ACL: (1) Foundation (standards body+IEEE), (2) Agent Communication Language (formal+standard), (3) Performatives (inform+request+query+propose), (4) Message structure (sender+receiver+content+conversation_id), (5) Ontology (shared vocab+semantics). Performatives: inform (tell fact+sender: informant), request (ask action+sender: requester), query (ask info+sender: asker), propose (suggest action+sender: proposer), accept-proposal (accept+sender: decider), reject-proposal (reject+sender: decider), call-for-proposal (CFP+sender: manager). ACLMessage: performative+sender+receiver+content+ontology+conversation_id (UUID)+reply_with (UUID)+timestamp+to_dict(). Criterios: FIPA ACL = standards+formal+IEEE, A2A = vendor+modern+Google, Custom = specific+internal+lightweight. Decision: standards -> FIPA, modern -> A2A, specific -> custom, mix -> ACL+A2A. Frameworks: fipa, langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + ACL.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar PERFORMATIVES con 7 performatives.
- Implementar ACLMessage con performative + sender + receiver + content.
- Implementar to_dict con conversation_id + reply_with.
- Diagnosticar performatives.
- Diagnosticar ACL vs A2A vs custom.

## Constrúyelo

```python
class ACLMessage:
    def __init__(self, performative, sender, receiver, content,
                 ontology=None, conversation_id=None, reply_with=None):
        self.performative = performative
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.ontology = ontology
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.reply_with = reply_with or str(uuid.uuid4())
        self.timestamp = time.time()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: fipa-acl-heritage
fase: 16
leccion: 02
---

1. 7 performatives.
2. ACLMessage.
3. to_dict.
4. +Production.
```

## Ejercicios

1. **Performatives**: probar
   los 7 performatives.
2. **ACLMessage**: probar
   to_dict.
3. **Desafio**: implementar
   un agente FIPA-compliant.

## Lecturas recomendadas

- "FIPA Specifications" (FIPA, 2002)
- "FIPA ACL Message Structure" (FIPA, 2002)
- "Multi-Agent Programming" (Bordini, 2009)

---

> 📚 **Adaptación al español de la lección [FIPA ACL Heritage]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).