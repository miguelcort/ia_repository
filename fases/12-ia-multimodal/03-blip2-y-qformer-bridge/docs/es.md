# BLIP-2 y Q-Former bridge

> BLIP-2 (Li 2023, Salesforce): frozen image encoder (EVA-CLIP ViT-G) + frozen LLM (FlanT5/OPT) + Q-Former bridge. Q-Former: 32 learnable queries (768 dim) con cross-attention a image features (ViT output, 257 tokens) + self-attention entre queries + FFN, output 32 image tokens. Linear projection 768 -> LLM_dim (4096). 2-stage training: Stage 1 vision-language (ITC, ITM, ITG), Stage 2 vision-language generation. +Efficient: -Parameters (Q-Former ~108M), -Compute, -Data. +Transfer: +SOTA 30+ datasets, +VQA, +captioning, +VLM. Variants: BLIP (encoder-decoder, ITC/ITM/ITG), BLIP-2 (Q-Former + frozen LLM), InstructBLIP (BLIP-2 + instruction tuning con 26 datasets + 13B params), BLIP-3 (2024, +video, +advanced). Frameworks: lavis (Salesforce), transformers (HF), vLLM. Hoy: BLIP-2 seminal, InstructBLIP +chat, BLIP-3 SOTA. Comparado: LLaVA +chat +open, Qwen-VL +multilingual, InternVL3 SOTA. Production: SigLIP encoder + LLaVA-Next chat.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/01, 12/02
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar qformer_queries.
- Implementar cross-attention.
- Apilar layers de Q-Former.
- Proyectar a LLM dim.
- Diagnosticar familia BLIP.

## Constrúyelo

```python
def qformer_forward(image_features, queries, W_q, W_kv, n_layers=2):
    h = queries
    for _ in range(n_layers):
        h = cross_attention(h, image_features, W_q, W_kv)
    return h
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: blip2-qformer
fase: 12
leccion: 03
---

1. Q-Former: 32 queries.
2. Cross-attn con image.
3. 2-stage training.
4. Frozen ViT + frozen LLM.
5. +Efficient, +transfer.
```

## Ejercicios

1. **Q-Former**: entrenar
   Q-Former con lavis.
2. **BLIP-2**: usar BLIP-2
   de HuggingFace.
3. **Desafio**: BLIP-2 para
   custom captioning.

## Lecturas recomendadas

- "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (Li et al., 2023)
- "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation" (Li et al., 2022)
- "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning" (Dai et al., 2023)
- "LAVIS: A Library for Language-Vision Intelligence" (Salesforce, 2023)

---

> 📚 **Adaptación al español de la lección [BLIP-2 Q-Former Bridge]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).