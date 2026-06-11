# LLaVA visual instruction tuning

> LLaVA (Liu 2023, Microsoft): linear projection ViT dim -> LLM dim (single matrix) + visual instruction tuning con 158K instances generadas por GPT-4 (58K conversation, 23K detail, 77K complex reasoning, formato USER/ASSISTANT con <image> placeholder). LLM backbones: Vicuna, Nous-Hermes, Mistral. +Open source, +chat, +SOTA 11 benchmarks. Variants: LLaVA (Vicuna-13B + CLIP-ViT-L-14, 158K), LLaVA-1.5 (Vicuna-13B + CLIP-ViT-L-336px, MLP projection 2 layers, 558K), LLaVA-Next (any resolution, dynamic, +SOTA), LLaVA-OneVision (single model, multi-image, video, +SOTA). Frameworks: llava, transformers (HF), vLLM, open_clip. Hoy: LLaVA-Next + LLaVA-OV SOTA. Comparado: LLaVA seminal, LLaVA-1.5 SOTA base, LLaVA-Next SOTA, LLaVA-OV video. Production: SigLIP encoder + LLaVA-Next chat + InternVL3 multilingual. Trade-offs: LLaVA + simple + open, LLaVA-Next SOTA, InternVL3 SOTA multilingual, Qwen-VL multilingual.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/01, 12/02, 11/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar clip_vit_forward.
- Implementar visual_projection.
- Construir conversation_format.
- Enmascarar targets para training.
- Diagnosticar familia LLaVA.

## Constrúyelo

```python
def visual_projection(image_features, W_proj):
    """Linear: ViT dim -> LLM dim."""
    return image_features @ W_proj
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: llava-instruct
fase: 12
leccion: 05
---

1. Linear ViT -> LLM.
2. 158K visual instructions.
3. GPT-4 generated.
4. Vicuna/Nous-Hermes.
5. +Open, +chat, +SOTA.
```

## Ejercicios

1. **LLaVA**: entrenar LLaVA
   con llava repository.
2. **LLaVA-Next**: usar
   LLaVA-Next de HuggingFace.
3. **Desafio**: LLaVA para
   custom VLM.

## Lecturas recomendadas

- "Visual Instruction Tuning" (Liu et al., 2023)
- "Improved Baselines with Visual Instruction Tuning" (Liu et al., 2023)
- "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge" (Liu et al., 2024)
- "LLaVA-OneVision: Easy Visual Task Transfer" (Li et al., 2024)

---

> 📚 **Adaptación al español de la lección [LLaVA Visual Instruction Tuning]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).