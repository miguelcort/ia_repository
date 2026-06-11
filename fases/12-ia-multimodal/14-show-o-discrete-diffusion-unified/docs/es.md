# Show-o discrete diffusion unified

> Show-o (2024): discrete diffusion unified con single transformer + text + image tokens en mismo vocab (VQ-VAE 8192 + text 50000 + special 128 = 58320 total) + masked token prediction (BERT-style: mask tokens random, predict masked, CE loss solo sobre masked positions) + iterative denoising sampling (start all masked, predict, replace masked, repeat). +Unified, +SOTA 2024-25, -VQ-VAE, -AR, -Continuous, +Discrete, +Simple. Variants: Show-o 1B, 7B. Frameworks: original, transformers (HF), vLLM. +Variants discrete diffusion: MDLM (Masked Diffusion LM), SEDD (Score Entropy Discrete Diffusion), Show-o. +Insights: -Continuous, +Discrete, +Simple, -VQ-VAE. Production: MIO + Show-o + TransFusion SOTA 2024-25. Familia: Show-o (discrete diffusion +simple -VQ-VAE), Emu3 (AR +fast VQ-VAE 32768), TransFusion (AR+continuous diffusion +quality shared transformer), MIO (any-to-any streaming +SOTA). Decision: simple -> Show-o, fast -> Emu3, quality -> TransFusion, any-to-any -> MIO. 2025: +Discrete diffusion + native + any-to-any + reasoning.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/11, 12/12, 12/13
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar show_o_vocab unificado.
- Implementar add_mask.
- Implementar show_o_denoise_step iterativo.
- Implementar show_o_loss sobre masked.
- Implementar show_o_sample_image.
- Diagnosticar discrete diffusion vs AR vs continuous diffusion.

## Constrúyelo

```python
def add_mask(tokens, mask_id, p_mask=0.5, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random(len(tokens)) < p_mask
    out = tokens.copy()
    out[mask] = mask_id
    return out, mask
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: show-o
fase: 12
leccion: 14
---

1. Discrete diffusion.
2. Single transformer.
3. Masked token prediction.
4. Iterative denoising.
5. +Unified +SOTA.
```

## Ejercicios

1. **Show-o**: usar Show-o
   con HuggingFace.
2. **MDLM**: entrenar
   MDLM en custom data.
3. **Desafio**: Show-o para
   multimodal generation.

## Lecturas recomendadas

- "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation" (Xie et al., 2024)
- "Simple Diffusion: End-to-End Diffusion for High Resolution Images" (Hoogeboom et al., 2023)
- "Score Entropy Discrete Diffusion" (Lou et al., 2024)
- "Masked Diffusion Language Models" (Shi et al., 2024)

---

> 📚 **Adaptación al español de la lección [Show-o Discrete Diffusion Unified]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).