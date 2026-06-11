# Walkthroughs de modelos open

> Walkthroughs de modelos SOTA open (2024-25): Llama 3 (Meta, 8B/70B/405B, GQA-8, BBPE 128K, RoPE base 500K, SwiGLU, RMSNorm, 8K/128K context con YaRN), Mistral 7B (GQA-4, sliding window 4096, BBPE 32K, RoPE theta 1M), Qwen 2.5 (Alibaba, 0.5B-72B, BBPE 152K multilingual, 128K context), Phi-3 (Microsoft, 3.8B/14B, MHA, BBPE 32K), Gemma 2 (Google, GQA-4, GeGLU, SentencePiece 256K), SmolLM (HuggingFace, 135M-1.7B). Componentes comunes: RMSNorm, RoPE, SwiGLU, GQA, BBPE. Licencias: Apache 2.0 (Llama 3, Mistral, Qwen, Phi-3), OpenRAIL (Gemma 2), custom.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/14-construye-un-transformer-capstone, 10/04-pre-training-mini-gpt
**Tiempo estimado:** ~30 minutos

## Objetivos

- Comparar arquitecturas de modelos open SOTA.
- Diagnosticar GQA, RMSNorm, RoPE, SwiGLU.
- Seleccionar modelo segun tamano, licencia, capacidad.
- Identificar trade-offs.

## Constrúyelo

```python
def llama3_architecture():
    return {
        "Params": "8B, 70B, 405B",
        "Attention": "GQA-8",
        "RoPE": "Base 500K",
        "FFN": "SwiGLU",
        "Norm": "RMSNorm",
        "Vocab": "128K BBPE",
        "Context": "8K (128K YaRN)",
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
name: prompt-open-models
fase: 10
leccion: 14
---

1. Llama 3, Mistral, Qwen 2.5.
2. Phi-3, Gemma 2, SmolLM.
3. GQA, RMSNorm, RoPE, SwiGLU.
4. BBPE 32K-152K vocab.
5. Apache 2.0 vs OpenRAIL.
```

## Ejercicios

1. **Llama 3**: deploy Llama 3 8B
   con vLLM y comparar.
2. **Mistral**: fine-tune Mistral 7B
   con TRL.
3. **Desafio**: implementar
   modelo from scratch estilo Llama 3.

## Lecturas recomendadas

- "The Llama 3 Herd of Models" (Dubey et al., 2024)
- "Mistral 7B" (Jiang et al., 2023)
- "Qwen Technical Report" (Bai et al., 2023)
- "Gemma: Open Models Based on Gemini Research and Technology" (Google DeepMind, 2024)

---

> 📚 **Adaptación al español** de la lección "[Open Models Architecture Walkthroughs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).