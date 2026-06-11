# CLIP contrastive pretraining

> CLIP (Radford 2021, OpenAI): dual encoder (image ViT/ResNet + text Transformer), contrastive learning con InfoNCE simetrica (N x N pairs, diagonal = positive, temperature scaling), zero-shot classification (cosine sim imagen vs text class names), entrenado en 400M image-text pairs (WIT). +Zero-shot, +transfer (30+ datasets), +robustness. Variantes SOTA: SigLIP (Google 2023, sigmoid loss, +efficient, +multilingual, WebLI 10B), ALIGN (Google), Florence (Microsoft), BLIP/BLIP-2 (Salesforce, +Q-Former + captioning), CoCa (Google, +captioner), InternVL (Shanghai AI Lab, +multilingual), OpenCLIP. SOTA production 2024-25: SigLIP (LLaVA-Next, Qwen-VL, InternVL), BLIP-2, LLaVA, InternVL3, Molmo. Frameworks: transformers (HF), open_clip, timm, vLLM. Hoy: SigLIP es SOTA production encoder, CLIP es seminal. Trade-offs: SigLIP + efficient + scale, CLIP seminal + simple, BLIP-2 + captioning, LLaVA + chat.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/09-vision-transformers, 12/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar l2_normalize y cosine_sim_matrix.
- Implementar clip_contrastive_loss (InfoNCE simetrica).
- Calcular top-k retrieval accuracy.
- Implementar zero-shot classification.
- Diagnosticar SigLIP y variantes SOTA.

## Constrúyelo

```python
def clip_contrastive_loss(image_emb, text_emb, temperature=0.07):
    logits = cosine_sim_matrix(image_emb, text_emb) / temperature
    n = logits.shape[0]
    labels = np.arange(n)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    log_softmax_image = np.log(e / e.sum(axis=-1, keepdims=True))
    loss_image = -log_softmax_image[np.arange(n), labels].mean()
    log_softmax_text = np.log(e / e.sum(axis=0, keepdims=True))
    loss_text = -log_softmax_text[np.arange(n), labels].mean()
    return 0.5 * (loss_image + loss_text), logits
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: clip-contrastive
fase: 12
leccion: 02
---

1. Dual encoder (ViT + Transformer).
2. InfoNCE simetrica.
3. Zero-shot classification.
4. 400M image-text pairs.
5. SigLIP SOTA production.
```

## Ejercicios

1. **CLIP**: entrenar CLIP
   con HuggingFace + imagenes.
2. **SigLIP**: usar SigLIP de
   transformers.
3. **Desafio**: zero-shot
   classification con CLIP.

## Lecturas recomendadas

- "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
- "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023)
- "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (Li et al., 2023)
- "OpenCLIP" (https://github.com/mlfoundations/open_clip)

---

> 📚 **Adaptación al español de la lección [CLIP Contrastive Pretraining]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).