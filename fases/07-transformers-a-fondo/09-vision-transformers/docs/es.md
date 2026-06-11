# Vision Transformers

> ViT: divide imagen en patches (16x16 o 14x14), linear projection, prepend [CLS], suma PE 2D, transformer encoder. Sin locality bias innato, necesita más datos que CNN. Variantes modernas: Swin (windowed, hierarchical), MAE (masked autoencoder), ConvNeXt (CNN modernizado), DINOv2 (self-supervised), SAM.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar image to patches.
- Implementar patch embeddings.
- Agregar [CLS] y PE 2D.
- Diagnosticar ViT vs CNN trade-offs.

## Constrúyelo

```python
def image_to_patches(img, patch_size):
    H, W, C = img.shape
    n_h, n_w = H // patch_size, W // patch_size
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patch = img[i*patch_size:(i+1)*patch_size,
                        j*patch_size:(j+1)*patch_size, :]
            patches.append(patch.flatten())
    return np.stack(patches)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vision-transformer
fase: 07
leccion: 09
---

1. ViT: patches -> linear -> [CLS] -> PE 2D.
2. Sin locality bias, data-hungry.
3. CLS head para classification.
4. Swin: windowed + hierarchical para detection.
5. MAE: 75% mask + decoder.
```

## Ejercicios

1. **Hybrid**: comparar ViT vs ResNet en small
   dataset (CIFAR-10).
2. **MAE**: implementar masked autoencoder para
   pre-training.
3. **Desafio**: implementar Swin windowed
   attention y comparar compute.

## Lecturas recomendadas

- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (Dosovitskiy et al., 2020)
- "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (Liu et al., 2021)
- "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)

---

> 📚 **Adaptación al español** de la lección "[Vision Transformers]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).