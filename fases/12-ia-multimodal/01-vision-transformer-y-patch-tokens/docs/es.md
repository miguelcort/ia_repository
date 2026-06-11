# Vision Transformer y patch tokens

> ViT (Dosovitskiy 2020, Google Brain): image split en patches (16x16 o 14x14), linear projection a d_model, [CLS] token prepend, positional embedding (learned o sinusoidal), standard transformer encoder, classification head sobre [CLS]. Architecture: ViT-B/16 (86M, 196 patches), ViT-L/16 (307M), ViT-H/14 (632M, 256 patches). -Inductive bias, +data hungry (JFT-300M, ImageNet-21k), +global attention desde layer 1. Variantes: Swin (Microsoft, windowed + hierarchical), BeiT (masked image modeling), MAE (He 2022, Meta, 75% mask), DINOv2 (Meta, self-supervised), ConvNeXt (CNN modernizada), EfficientFormer, EVA-02, SigLIP, InternImage. SOTA 2024-25 vision encoders: SigLIP (Google, sigmoid loss + scale + multilingual), EVA-02 (BAAI, +accuracy, MIM + CLIP), DINOv2, InternVL3. Multimodal: LLaVA, Qwen-VL, InternVL3 (4B-72B), Molmo, LLaVA-Next. Frameworks: timm, transformers, torchvision. Hoy: SigLIP + EVA-02 + LLaVA-Next + InternVL3 es SOTA production. Frontier: vision + reasoning (o1-style) + video + 3D + document.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/09-vision-transformers
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar image_to_patches.
- Implementar patch embedding.
- Agregar [CLS] y positional embedding.
- Calcular numero de patches.
- Diagnosticar variantes y trade-offs.

## Constrúyelo

```python
def image_to_patches(img, patch_size=16):
    H, W, C = img.shape
    n_h, n_w = H // patch_size, W // patch_size
    return img[:n_h*patch_size, :n_w*patch_size, :].reshape(
        n_h, patch_size, n_w, patch_size, C).transpose(0, 2, 1, 3, 4).reshape(n_h * n_w, -1)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: vision-transformer
fase: 12
leccion: 01
---

1. Image -> patches -> linear.
2. + CLS + PE -> transformer.
3. Swin, BeiT, MAE, DINOv2.
4. SigLIP, EVA-02 SOTA.
5. Multimodal: LLaVA, Qwen-VL.
```

## Ejercicios

1. **ViT**: implementar ViT
   forward en numpy.
2. **Swin**: implementar
   windowed attention.
3. **Desafio**: ViT para
   custom dataset.

## Lecturas recomendadas

- "An Image is Worth 16x16 Words: ViT" (Dosovitskiy et al., 2020)
- "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., 2021)
- "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)
- "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)

---

> 📚 **Adaptación al español de la lección [Vision Transformer Patch Tokens]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).