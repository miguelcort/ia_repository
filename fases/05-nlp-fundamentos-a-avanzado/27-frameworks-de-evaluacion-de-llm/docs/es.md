# Frameworks de evaluación de LLM

> Evaluar LLMs requiere frameworks especificos: lm-eval-harness (HuggingFace, 200+ benchmarks), RAGAS (RAG), DeepEval (custom), LangSmith (production), TruLens. Metricas: EM, F1, BLEU, ROUGE, faithfulness, LLM-as-judge. Human eval sigue siendo gold standard.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 26-extraccion-de-relaciones-y-grafo-de-conocimiento
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar EM, F1 token, ROUGE-L.
- Implementar faithfulness para RAG.
- Implementar LLM-as-judge mock.
- Diagnosticar sesgos del LLM-as-judge.

## Constrúyelo

```python
def faithfulness_score(respuesta, contexto):
    r = set(respuesta.lower().split())
    c = set(contexto.lower().split())
    if not r: return 0.0
    return len(r & c) / len(r)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-llm-eval
fase: 05
leccion: 27
---

1. Publicos: lm-eval-harness, HELM, BIG-Bench.
2. RAG: RAGAS (faithfulness, relevance, precision, recall).
3. Custom: DeepEval, Promptfoo, LangSmith.
4. Production: shadow + A/B + human eval.
5. Safety: Detoxify, Llama Guard.
6. LLM-as-judge con multi-judge y rubric.
```

## Ejercicios

1. **lm-eval-harness**: correr MMLU y HellaSwag en
   HuggingFace.
2. **RAGAS**: implementar eval completo de RAG con tu
   dataset.
3. **Desafio**: framework de eval production con golden
   set, LLM-as-judge multi-modelo, human eval
   periodica, reportes en CI.

## Lecturas recomendadas

- "lm-eval-harness" (Gao et al., 2024)
- "RAGAS" (Es et al., 2023)
- "HELM" (Liang et al., 2022)

---

> 📚 **Adaptación al español** de la lección "[LLM Evaluation Frameworks]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).