# Chatbots: de reglas a neuronal

> La evolucion: ELIZA (1966) -> FAQ matching -> seq2seq -> RAG -> LLMs con tools y memoria. Hoy: Claude/GPT + RAG + function calling + guardrails. Frameworks: LangChain, LlamaIndex, Rasa, BotPress.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16-generacion-de-texto-pre-transformer
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar chatbot de reglas.
- Implementar retrieval chatbot (FAQ matching).
- Implementar generativo mock.
- Diagnosticar cuando usar cada enfoque.

## Constrúyelo

```python
def chatbot_retrieval(pregunta, faq):
    q_tokens = set(re.findall(r"\b\w+\b", pregunta.lower()))
    mejor = None
    mejor_score = 0
    for q_faq, respuesta in faq.items():
        f_tokens = set(re.findall(r"\b\w+\b", q_faq.lower()))
        overlap = len(q_tokens & f_tokens) / len(q_tokens | f_tokens)
        if overlap > mejor_score:
            mejor_score = overlap
            mejor = respuesta
    return mejor, mejor_score
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-chatbot
fase: 05
leccion: 17
---

1. FAQ: Rasa, BotPress.
2. Retrieval: RAG + LLM.
3. Agente: LangChain / LlamaIndex.
4. Memoria: LangChain memory, Zep.
5. Self-host: Llama 3 + vLLM.
6. Persona + guardrails + streaming + fallback.
```

## Ejercicios

1. **Intents**: entrenar un clasificador de intents
   (e.g. BERT fine-tune).
2. **Dialog state**: implementar maquina de estados para
   un booking flow.
3. **Desafio**: chatbot production con RAG (Chroma +
   sentence-transformers + Claude API) y memory de 10
   turnos, deploy en FastAPI.

## Lecturas recomendadas

- "ELIZA" (Weizenbaum, 1966) — historico
- "Neural Conversational Model" (Vinyals & Le, 2015)
- LangChain: <https://python.langchain.com/>
- LlamaIndex: <https://www.llamaindex.com/>

---

> 📚 **Adaptación al español** de la lección "[Chatbots: Rule to Neural]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).