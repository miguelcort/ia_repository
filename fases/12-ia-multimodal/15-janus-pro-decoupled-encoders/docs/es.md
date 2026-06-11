# Janus-Pro decoupled encoders

> Janus-Pro (DeepSeek 2025): decoupled encoders - SigLIP encoder para understanding (1152 dim, 257 tokens) + VQ tokenizer para generation (32768 codes) + single transformer unified. +Specialized, +Quality, +SOTA 2024-25. Variants: Janus 1B, Janus-Pro 1B, 7B, JanusFlow. Frameworks: original, transformers (HF), vLLM, SGLang, open_clip. +Insights: -Encoder unico, +Specialized, +Quality, -Simple. Por que decoupled: SigLIP para understanding (+OCR +Detail +Semantic +Transfer), VQ para generation (+Discrete +AR +Simple +Fast), +Specialized, -Compromise. Comparado: Emu3 (single AR + VQ-VAE 32768 +Simple), Show-o (discrete diffusion +Simple -VQ-VAE), Chameleon (single VQ-VAE 8192 seminal), MIO (any-to-any streaming +SOTA 2024-25). Decision: specialized -> Janus-Pro, simple -> Emu3, diffusion -> Show-o, any-to-any -> MIO. Production: MIO + Janus-Pro + TransFusion + Show-o SOTA. 2025: +Decoupled + specialized + native + reasoning.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/02, 12/10, 12/11, 12/12
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar janus_understanding_encoder (SigLIP).
- Implementar janus_generation_encoder (VQ).
- Implementar janus_route_modality.
- Implementar janus_forward dual mode.
- Diagnosticar decoupled vs single encoder.

## Constrúyelo

```python
def janus_understanding_encoder(image, embed_dim=1152):
    rng = np.random.default_rng(hash(image.tobytes()[:64]) & 0xFFFFFFFF)
    H, W, C = image.shape
    n = (H // 14) * (W // 14) + 1
    return rng.standard_normal((n, embed_dim)) * 0.1
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: janus-pro
fase: 12
leccion: 15
---

1. Decoupled encoders.
2. SigLIP + VQ.
3. Single transformer.
4. +Specialized.
5. +SOTA 2025.
```

## Ejercicios

1. **Janus-Pro**: usar
   Janus-Pro con HuggingFace.
2. **SigLIP**: entrenar
   SigLIP en custom data.
3. **Desafio**: Janus-Pro
   para understanding + gen.

## Lecturas recomendadas

- "Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Engineering" (Chen et al., 2025)
- "Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation" (Ma et al., 2024)
- "DeepSeek-VL: Towards Real-World Vision-Language Understanding" (Lu et al., 2024)
- "JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation" (Ma et al., 2024)

---

> 📚 **Adaptación al español de la lección [Janus-Pro Decoupled Encoders]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).