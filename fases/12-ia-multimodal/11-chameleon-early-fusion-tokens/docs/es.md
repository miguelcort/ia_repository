# Chameleon early fusion tokens

> Chameleon (Meta 2024): early fusion con image tokens discretizados via VQ-VAE codes (codebook 8192, patch 16x16) + text + image en mismo vocabulario (text 32000 + image 8192 + special 128 = 40320 total) + single transformer autoregresivo. +Unified, +SOTA, +generation. Variants: Chameleon 7B, 34B. +Insights: -Dual encoder, -Cross-attn, +Simple. Frameworks: original, transformers (HF), vLLM. +Production: Chameleon seminal. +Use cases: multimodal, image generation, text generation. VQ-VAE: encoder image -> latent -> quantize (nearest codebook) -> discrete id -> decoder codebook id -> image. -Reconstruction error, +Discrete, +Compression. Fusion strategies: late (LLaVA, BLIP-2, Flamingo, Qwen-VL: vision encoder + LLM, +simple, +efficient, +SOTA) vs early (Chameleon, MIO, TransFusion: image + text en mismo espacio, +unified, +generation, -compute). Production: late (LLaVA-Next, InternVL3, Qwen2.5-VL). Research: early (Chameleon, MIO, Emu3, TransFusion, Show-o).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/01, 12/02
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar vqvae_encode y vqvae_decode.
- Implementar chameleon_vocab unificado.
- Encode text y image tokens.
- Concatenar en chameleon_sequence.
- Diagnosticar fusion strategies.

## Constrúyelo

```python
def chameleon_vocab(vocab_size_text=32000, vocab_size_image=8192, special_tokens=128):
    return vocab_size_text + vocab_size_image + special_tokens
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: chameleon-early
fase: 12
leccion: 11
---

1. Early fusion.
2. VQ-VAE 8192.
3. Vocab unificado 40320.
4. Single transformer.
5. +Unified, +generation.
```

## Ejercicios

1. **Chameleon**: usar
   Chameleon con HuggingFace.
2. **VQ-VAE**: entrenar
   VQ-VAE en custom data.
3. **Desafio**: Chameleon
   para image generation.

## Lecturas recomendadas

- "Chameleon: Mixed-Modal Early-Fusion Foundation Models" (Team Chameleon, 2024)
- "Neural Discrete Representation Learning" (Van den Oord et al., 2017)
- "MIO: A Foundation Model on Multimodal Tokens" (Wang et al., 2024)
- "Emu3: Next-Token Prediction is All You Need" (Wang et al., 2024)

---

> 📚 **Adaptación al español de la lección [Chameleon Early Fusion Tokens]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).