# Flamingo gated cross-attention

> Flamingo (Alayrac 2022, DeepMind): few-shot multimodal in-context learning con frozen Perceiver Resampler (comprime variable-length image features a 64 tokens via cross-attn con 64 learnable latents) + gated cross-attention layers inserted entre LLM blocks (cross-attn LLM text con vision features, output = text_h + tanh(alpha) * attn, alpha aprendible inicializado a 0). +Init a 0: -Impact training inicial, +stable, aprende progresivamente. +Few-shot: image-text pairs en prompt, +transfer, +SOTA 16 datasets. Variants: Flamingo, OpenFlamingo, Otter, Flamingo + LLaVA. Frameworks: open_flamingo, transformers (HF), vLLM. Hoy: Flamingo seminal, +VLM modern (LLaVA, Qwen-VL, InternVL3). +Use cases: captioning, VQA, multimodal chat. Trade-offs: Flamingo + few-shot + Perceiver - compute, BLIP-2 + efficient + Q-Former - params, LLaVA + chat + instruction tuned + open. Production: SigLIP encoder + LLaVA-Next chat.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/01, 12/02, 12/03
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar gated_cross_attention con tanh gate.
- Apilar flamingo_block.
- Alternar cross-attn con multiple vision sets.
- Diagnosticar Perceiver Resampler.
- Diagnosticar trade-offs Flamingo vs BLIP-2 vs LLaVA.

## Constrúyelo

```python
def gated_cross_attention(text_h, vision_kv, W_q, W_kv, W_o, gate_logit_zero_init=True):
    q = text_h @ W_q
    k = vision_kv @ W_kv.T
    v = vision_kv @ W_kv.T
    out = (softmax(q @ k.T / sqrt(d)) @ v) @ W_o
    gate = 0.0 if gate_logit_zero_init else 1.0
    return text_h + np.tanh(gate) * out
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: flamingo-gated
fase: 12
leccion: 04
---

1. Perceiver Resampler.
2. Gated cross-attn.
3. tanh gate init 0.
4. Few-shot multimodal.
5. +16 datasets SOTA.
```

## Ejercicios

1. **Flamingo**: entrenar
   Flamingo con open_flamingo.
2. **OpenFlamingo**: few-shot
   inference con OpenFlamingo.
3. **Desafio**: Flamingo para
   custom VLM.

## Lecturas recomendadas

- "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al., 2022)
- "Perceiver: General Perception with Iterative Attention" (Jaegle et al., 2021)
- "OpenFlamingo: An Open-Source Framework for Training Few-Shot Vision-Language Models" (Awadalla et al., 2023)

---

> 📚 **Adaptación al español de la lección [Flamingo Gated Cross Attention]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).