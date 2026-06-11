# Pipeline completo de LLM

> Pipeline end-to-end LLM: data (FineWeb, RedPajama) → pre-training (10-15T tokens, AdamW, cosine, 1000s GPUs, $50-500M) → SFT (Alpaca, Magpie, 10M+ examples) → DPO/RLHF (UltraFeedback, iterative) → eval (MMLU, HumanEval, GPQA, Arena) → quantization (INT4 AWQ, 4x reduction, 50+ tok/sec) → inference (vLLM/SGLang, paged attention, prefix caching) → deployment (cloud, edge, mobile, browser). Frontier: Llama 3 70B Instruct, GPT-4, Claude 3.5, Gemini 1.5, Mistral Large, Qwen 2.5.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 10/04-pre-training-mini-gpt, 10/06-instruction-tuning-sft, 10/08-dpo, 10/11-cuantizacion, 10/12-optimizacion-de-inferencia
**Tiempo estimado:** ~30 minutos

## Objetivos

- Listar las 8 stages del pipeline.
- Calcular pre-training cost.
- Diagnosticar quantization trade-offs.
- Comparar deployment options.

## Constrúyelo

```python
def pre_train_cost(n_params, n_tokens):
    return 6 * n_params * n_tokens
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-llm-pipeline
fase: 10
leccion: 13
---

1. Data -> pre-train -> SFT -> DPO/RLHF.
2. Eval -> quant -> inference.
3. Llama 70B $50-100M, 405B $200-500M.
4. INT4 AWQ 4x reduction.
5. Cloud, edge, mobile, browser.
```

## Ejercicios

1. **Pipeline**: implementar pipeline
   end-to-end en custom data.
2. **Cost**: estimar cost de
   entrenar 7B con Chinchilla.
3. **Desafio**: deploy Llama 3 70B
   con vLLM + AWQ.

## Lecturas recomendadas

- "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023)
- "The Llama 3 Herd of Models" (Dubey et al., 2024)
- "Hugging Face Transformers" (Wolf et al., 2020)
- "vLLM: Efficient Memory Management for LLM Serving" (Kwon et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[Building Complete LLM Pipeline]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).