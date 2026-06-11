# Evaluación de LLM apps

> A/B testing (Statsig, Eppo, Optimizely, PostHog), human eval (raters 1-5, +expensive gold standard), LM-as-judge (GPT-4 judge, AlpacaEval, MT-Bench, 70-90% agreement con humans, biases: self-bias, length bias, position bias). Frameworks: RAGAS (RAG faithfulness, relevance, recall), promptfoo (LLM eval + prompt versioning), Braintrust, DeepEval, TruLens, LangSmith, Phoenix, RAGChecker. LLM judges SOTA: Prometheus 2 (open), JudgeLRM, PandaLM. Metrics: precision/recall, BLEU, faithfulness. Continuous monitoring: real-time metrics + alerts. SOTA 2024-25: multi-metric (RAGAS + LM-judge + human + A/B + benchmarks) + continuous monitoring + agentic eval. Benchmarks: MMLU, HumanEval, GPQA, MATH, MMLU-Pro, FrontierMath, HLE, LiveCodeBench, RGB, BEIR.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 11/06-rag, 11/08-fine-tuning-lora
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar LM-as-judge score.
- Implementar A/B test significance.
- Implementar precision/recall.
- Implementar BLEU score.
- Diagnosticar frameworks SOTA.

## Constrúyelo

```python
def lm_judge_score(prompt, response, reference):
    if not response or not reference:
        return 0.0
    r_tokens = set(response.split())
    ref_tokens = set(reference.split())
    if not ref_tokens: return 0.0
    return min(1.0, len(r_tokens & ref_tokens) / len(ref_tokens))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: llm-evaluation
fase: 11
leccion: 10
---

1. A/B tests, human eval, LM-judge.
2. RAGAS, promptfoo, Braintrust.
3. Prometheus 2, JudgeLRM.
4. Statsig, Eppo A/B.
5. Continuous monitoring.
```

## Ejercicios

1. **A/B test**: implementar A/B
   test pipeline.
2. **RAGAS**: eval RAG system
   con RAGAS.
3. **Desafio**: LM-judge pipeline
   production.

## Lecturas recomendadas

- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023)
- "RAGAS: Automated Evaluation of Retrieval Augmented Generation" (Es et al., 2023)
- "Prometheus 2: An Open and Specialized LLM for Evaluation-as-a-Service" (Kim et al., 2024)
- "Braintrust: Evals for LLM Apps" (Anyscale, 2024)

---

> 📚 **Adaptación al español** de la lección "[Evaluation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).