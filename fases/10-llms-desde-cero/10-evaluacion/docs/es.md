# Evaluación de LLMs

> Benchmarks SOTA: MMLU (57 subjects, multitask), HellaSwag, ARC, GSM8K (math), HumanEval/MBPP (code), TruthfulQA, MMLU-Pro, GPQA Diamond (graduate). Frameworks: lm-evaluation-harness (EleutherAI, +100 benchmarks), HELM, AlpacaEval, OpenAI evals. LM-as-judge (GPT-4 judge, +80% agreement con humans, pero self-bias y length bias). LMSYS Chatbot Arena: human preference Elo, +1M votes. Code eval: Pass@k = 1 - C(n-c,k)/C(n,k), SWE-bench, LiveCodeBench. Limitaciones: contamination, saturation, narrow tasks, format dependence.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/04-pre-training-mini-gpt
**Tiempo estimado:** ~30 minutos

## Objetivos

- Listar benchmarks SOTA.
- Implementar LM-as-judge mock.
- Calcular Pass@k para code.
- Implementar Elo rating para arena.
- Diagnosticar frameworks.

## Constrúyelo

```python
def pass_at_k(n_samples, n_correct, k):
    return 1 - comb(n - n_correct, k) / comb(n, k)
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
fase: 10
leccion: 10
---

1. MMLU, HumanEval, GPQA, etc.
2. LM-as-judge, position bias.
3. LMSYS Arena, Elo.
4. Pass@k code eval.
5. lm-eval-harness.
```

## Ejercicios

1. **lm-eval-harness**: correr
   benchmark suite.
2. **Arena**: simular A/B voting
   entre modelos.
3. **Desafio**: implementar
   custom benchmark.

## Ejercicios

1. **lm-eval-harness**: correr benchmark suite en
   un modelo HF (TinyLlama 1.1B).
2. **Arena**: simular A/B voting entre dos modelos,
   calcular Elo.
3. **Pass@k**: escribir generador + tests para
   HumanEval problem 0 (sumar dos números).
4. **Desafío**: implementar LLM-as-judge usando
   Claude API y evaluar 50 outputs propios.

## Limitaciones y frontier benchmarks

Limitaciones críticas: (1) Contamination: el test set
puede estar en training data. Mitigación: decontaminate
con n-gram overlap, dynamic benchmarks (LiveCodeBench,
ARC-AGI). (2) Saturation: 90%+ en MMLU, HumanEval; ya
no discrimina entre frontier models. (3) Format
dependence: 5-shot CoT vs 0-shot cambia 10-30 puntos.
(4) Linguistic bias: English-centric.

Frontier benchmarks 2024-2026: (1) FrontierMath (Epoch
AI, 100 problemas, +hard). (2) Humanity's Last Exam
(HLE, multi-domain). (3) ARC-AGI (Abstraction
Reasoning Corpus). (4) GPQA Diamond (198 graduate
q&a). (5) MMLU-Pro (12K, harder). (6) BigCodeBench
(real code, 1000+). (7) WildBench, LiveBench (real
prompts).

Frameworks: lm-evaluation-harness (EleutherAI, +100
benchmarks, el standard). HELM (Stanford, holistic,
multi-metric). AlpacaEval (LLM judge, MT-Bench,
+rápido). HELM Safety, DecodingTrust (safety).
OpenLLM Leaderboard v2. OpenAI evals framework.
LMSYS Arena (real human preference). HuggingFace
evaluate (Python library, glue, superglue).

## Lecturas recomendadas

- "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2020)
- "Evaluating the Unevaluable: HumanEval" (Chen et al., 2021)
- "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023)
- "lm-evaluation-harness" (EleutherAI, Gao et al., 2021)
- "FrontierMath" (Epoch AI, 2024)
- "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (Jimenez et al., 2024)
- "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code" (2024)

---

> 📚 **Adaptación al español** de la lección "[Evaluation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).