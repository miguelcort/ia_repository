# 68 — RAG eval: precision, recall, faithfulness

> RAG eval: context_precision (relevant / retrieved), context_recall (retrieved / relevant), faithfulness (no hallucinations), answer_relevancy. Frameworks: RAGAS, TruLens, deep_eval.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** Fase 19/08
**Tiempo estimado:** ~25 minutos

## Objetivos

- Context precision/recall.
- Faithfulness.
- Answer relevancy.
- RAGAS suite.

## Constrúyelo

```python
def context_precision(retrieved, relevant):
    """K/R: relevant docs / retrieved."""
    return len(set(retrieved) & set(relevant)) / max(
        len(retrieved), 1)


def faithfulness(answer, context):
    """LLM judge: answer solo usa context?"""
    return llm_judge(f"Context: {context}\nAnswer: {answer}\n"
                    f"Is the answer fully supported?")


def rag_eval(rag_system, test_set):
    results = {"precision": [], "recall": [], "faithful": []}
    for item in test_set:
        retrieved = rag_system.retrieve(item["question"])
        answer = rag_system.answer(item["question"])
        results["precision"].append(context_precision(
            retrieved, item["relevant"]))
        results["recall"].append(context_recall(
            retrieved, item["relevant"]))
        results["faithful"].append(faithfulness(answer,
                                                retrieved))
    return {k: sum(v) / len(v) for k, v in results.items()}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-rag-eval
fase: 19
leccion: 68
---

1. Context precision/recall.
2. Faithfulness.
3. Answer relevancy.
4. RAGAS.
```

## Ejercicios

1. **RAGAS**: 100 q&a
   dataset.
2. **Faithfulness**: GPT-4
   judge.
3. **Desafío**: +0.1
   faithfulness.

## Lecturas recomendadas

- "RAGAS" (Es 2023)
- "TruLens" (2024)
- "deep_eval" (Confident AI)

---

> 📚 **Adaptación al español** de la lección
> "[68-rag-eval-precision-recall]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
