# Preguntas y respuestas

> Dada una pregunta y un contexto, encontrar o generar la respuesta. SOTA: BERT fine-tune en SQuAD (~95% F1), RAG con LLM para open-domain. Frameworks: Haystack, LlamaIndex, LangChain.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12-resumen-de-texto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar QA extractivo simple con overlap de tokens.
- Calcular EM y F1 a nivel de tokens.
- Diagnosticar closed vs open-domain.
- Diagnosticar RAG.

## Constrúyelo

```python
class SimpleQASystem:
    def predecir(self, pregunta, top_k=1):
        q_tokens = set(tokenizar(pregunta)) - stopwords
        for ctx in self.contextos:
            for oracion in re.split(r'(?<=[.!?])\s+', ctx):
                o_tokens = set(tokenizar(oracion)) - stopwords
                score = len(q_tokens & o_tokens) / len(q_tokens)
                ...
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-qa
fase: 05
leccion: 13
---

1. Closed custom: BERT SQuAD fine-tune o RAG.
2. Open-domain: BM25/DPR + FiD o RAG + LLM.
3. Multi-hop: HopQA, ReAct.
4. Long-form: LLM con contexto.
5. RAG: chunking + ANN + reranker + citations.
```

## Ejercicios

1. **DPR retriever**: implementar dense passage retrieval con
   bi-encoder.
2. **SQuAD eval**: implementar EM y F1 segun la definicion
   oficial.
3. **Desafio**: RAG pipeline completo con chunking,
   embeddings, FAISS, Claude y citations, evaluar con
   RAGAS.

## Lecturas recomendadas

- "SQuAD" (Rajpurkar et al., 2016)
- "DPR" (Karpukhin et al., 2020)
- "RAG" (Lewis et al., 2020)
- Haystack: <https://haystack.deepset.ai/>

---

> 📚 **Adaptación al español** de la lección "[Question Answering]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).