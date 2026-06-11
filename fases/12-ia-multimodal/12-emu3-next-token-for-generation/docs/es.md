# Emu3 next token for generation

> Emu3 (Wang 2024, BAAI): next-token prediction is all you need - unifies understanding + generation en single token space (VQ-VAE 32768 + text 100352 + special 128 = 133248 total vocab). Image -> VQ codes (patch 16x16, 32768 codes), text -> token ids (100352 BPE), single transformer autoregresivo (text->text, image->image, text->image, image->text). +SOTA 2024-25 unified, -Dual architecture, -Cross-attn, -Projection, +Simple, +Transfer. Variants: Emu3 8B, 12B. Frameworks: original, transformers (HF), vLLM. +Production: Emu3 seminal. +Use cases: multimodal, generation, any-to-any. Familia unified multimodal: Emu3 (BAAI 2024 next-token VQ-VAE 32768), Chameleon (Meta 2024 early fusion VQ-VAE 8192 7B-34B), MIO (2024 any-to-any streaming +SOTA), TransFusion (2024 AR+diffusion +SOTA), Show-o (2024 discrete diffusion unified +SOTA). Production: MIO + TransFusion + Show-o SOTA 2024-25. Frameworks: original, transformers, vLLM, SGLang. Trade-offs: Emu3 + simple, MIO + SOTA, TransFusion + AR+diffusion, Show-o + diffusion. 2025: +Native + any-to-any + reasoning + video.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/11
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar emu3_tokenizer unificado.
- Implementar emu3_image_tokens (VQ).
- Implementar emu3_understand (concat).
- Implementar emu3_generate autoregresivo.
- Diagnosticar familia unified multimodal.

## Constrúyelo

```python
def emu3_tokenizer(visual_codebook_size=32768, text_vocab_size=100352, special_tokens=128):
    return visual_codebook_size + text_vocab_size + special_tokens
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: emu3
fase: 12
leccion: 12
---

1. Next-token prediction.
2. Single vocab 133248.
3. VQ-VAE 32768.
4. Autoregresivo.
5. +Unified +SOTA.
```

## Ejercicios

1. **Emu3**: usar Emu3
   con HuggingFace.
2. **VQ-VAE**: entrenar
   VQ-VAE en custom data.
3. **Desafio**: Emu3 para
   image generation.

## Lecturas recomendadas

- "Emu3: Next-Token Prediction is All You Need" (Wang et al., 2024)
- "Chameleon: Mixed-Modal Early-Fusion Foundation Models" (Team Chameleon, 2024)
- "MIO: A Foundation Model on Multimodal Tokens" (Wang et al., 2024)
- "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model" (Zhou et al., 2024)

---

> 📚 **Adaptación al español de la lección [Emu3 Next Token for Generation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).