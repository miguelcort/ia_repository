# InternVL3 native multimodal

> InternVL3 (Chen 2024, Shanghai AI Lab): native multimodal pretraining con InternViT-300M/6B vision encoder + Qwen2.5 LLM backbone + dynamic resolution 1-12 tiles (aspect preservado) + pixel shuffle (spatial downsample, channel upsample, -75% tokens con scale=2) + MLP projector (2-3 layers). 1B-72B params. +SOTA 2024-25 multilingual. Variants: InternVL 1.0 (seminal, +OCR), InternVL 1.5 (+SOTA +multilingual Qwen), InternVL 2.0 (+SOTA +multilingual +dynamic Qwen2), InternVL3 (native + SOTA + pixel shuffle + Qwen2.5). Frameworks: transformers (HF), vLLM, SGLang, open_clip. Pixel shuffle: (n, d) -> (n/scale^2, d*scale^2) -75% tokens, -75% attention compute, +efficiency. Production: InternVL3 SOTA. +Use cases: VLM, OCR, multilingual, document, video. +Trade-offs: pixel shuffle + efficiency, projection + simple. SOTA 2024-25 mixto: InternVL3 + Qwen2.5-VL + Molmo + LLaVA-Next.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/06, 12/09
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar intern_vit_forward.
- Implementar intern_dynamic_tile (1-12 tiles).
- Implementar pixel_shuffle_reshape (-75% tokens).
- Implementar intern_mlp_projector.
- Implementar internvl3_forward completo.

## Constrúyelo

```python
def pixel_shuffle_reshape(features, scale=2):
    n, d = features.shape
    n_new = n // (scale * scale)
    d_new = d * (scale * scale)
    return features[:n_new*scale*scale].reshape(n_new, scale*scale, d).reshape(n_new, d*scale*scale)[:, :d_new]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: internvl3
fase: 12
leccion: 10
---

1. InternViT + Qwen2.5.
2. Native multimodal.
3. Dynamic 1-12 tiles.
4. Pixel shuffle -75% tokens.
5. +SOTA multilingual.
```

## Ejercicios

1. **InternVL3**: usar
   InternVL3 con HuggingFace.
2. **Pixel shuffle**: entrenar
   con pixel shuffle.
3. **Desafio**: InternVL3
   para custom VLM.

## Lecturas recomendadas

- "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks" (Chen et al., 2024)
- "InternVL2: Better than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models" (Chen et al., 2024)
- "InternVL3: Exploring Native Multimodal Large Models" (Chen et al., 2024)

---

> 📚 **Adaptación al español de la lección [InternVL3 Native Multimodal]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).