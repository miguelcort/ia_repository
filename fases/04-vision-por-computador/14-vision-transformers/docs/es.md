# 14 — Vision Transformers (ViT)

> ViT aplica transformers directamente a patches de imagen. Supera a las CNNs en clasificación a escala, dominando el benchmark ImageNet desde 2020.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07-transformers-a-fondo
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar ViT desde cero con PyTorch.
- Aplicar patch embedding, positional encoding y CLS token.
- Diagnosticar cuándo ViT supera a CNN y viceversa.
- Conocer variantes: Swin, DeiT, MaxViT, ConvNeXt.

## El problema

Las CNNs son dominantes en visión desde 2012. En 2020,
Dosovitskiy et al. demostraron que un **transformer puro**
(sin convoluciones) puede igualar y superar a las CNNs
state-of-the-art en clasificación, siempre que se entrene
con suficientes datos. ViT divide la imagen en patches
fijos, los proyecta linealmente, y aplica un transformer
encoder estándar. La lección cubre la arquitectura y
cuándo usarla.

## El concepto

**Patch embedding.** Divide la imagen en patches fijos
(e.g. 16x16 para ViT-B). Cada patch se aplana y se proyecta
linealmente a un vector de dimensión `D`. Si la imagen es
224x224 con patches 16x16, hay 196 patches.

**Positional encoding.** Como el transformer es
permutation-invariant, hay que añadir información
posicional. ViT usa **positional embeddings aprendibles**:
un vector por patch. Se añaden a los patch embeddings.

**CLS token.** Un token adicional (aprendible) que
agregamos al inicio de la secuencia. Tras la atención, su
representación se usa para clasificación (vía un MLP
head). Equivalente al [CLS] de BERT.

**Arquitectura ViT.** Pre-norm transformer: `LayerNorm →
Multi-Head Attention → residual → LayerNorm → MLP →
residual`. `L` bloques. MLP head al final sobre el CLS
token.

**Variantes de ViT.**

- **DeiT (Data-efficient Image Transformer):** añade un
  teacher CNN para destilación. ViT funciona con menos
  datos.
- **Swin Transformer:** atención en ventanas
  (shifted windows). Jerárquica como CNN, eficiente.
- **MaxViT:** combina convs locales con atención global.
  Eficiente y SOTA.
- **ConvNeXt:** modernizar ResNet con ideas de ViT
  (LayerNorm, GELU, depthwise convs grandes).
- **EVA, EVA-02:** ViTs a gran escala preentrenados con
  masked image modeling. Estado del arte 2023-2024.

**Cuándo usar ViT vs CNN.**

| Aspecto | CNN | ViT |
|---|---|---|
| Datos pequeños (< 10k) | Mejor | Peor (overfit) |
| Datos grandes (> 100k) | Bueno | Mejor |
| Inductive bias (localidad) | Sí | No |
| Eficiencia | Alta | Media |
| Interpretabilidad | Filtros aprendidos | Attention maps |

**Hiperparámetros ViT.**

- `image_size`: 224, 384, 512.
- `patch_size`: 16 (estándar), 14, 8 (más fino, más
  tokens).
- `dim`: 768 (ViT-B), 1024 (ViT-L), 1280 (ViT-H).
- `depth`: 12 (B), 24 (L), 32 (H).
- `heads`: 12, 16, 16.
- `mlp_dim`: 4*dim.

**Trampas.**

- **Sin suficientes datos:** ViT no generaliza. Usa
  augmentations fuertes o transfer learning.
- **Patch size mal elegido:** patches grandes pierden
  detalle local; patches pequeños explotan la
  complejidad.
- **Sin preentrenamiento:** ViT necesita preentrenarse
  en datasets grandes (ImageNet-21k, JFT-300M).

## Constrúyelo

```python
import numpy as np


def patchify(im, patch_size=16):
    """Divide imagen (H, W, C) en patches (N, patch_size, patch_size, C)."""
    H, W, C = im.shape
    assert H % patch_size == 0 and W % patch_size == 0
    n_h = H // patch_size
    n_w = W // patch_size
    patches = im.reshape(n_h, patch_size, n_w, patch_size, C)
    patches = patches.transpose(0, 2, 1, 3, 4)
    return patches.reshape(n_h * n_w, patch_size * patch_size * C)


def patch_embed(patches, dim=768):
    """Proyecta patches a dim. patches: (N, P*P*C), dim: D."""
    return patches @ np.random.randn(patches.shape[-1], dim) * 0.02


def positional_encoding_2d(n_patches, dim):
    """Positional embeddings aprendibles (inicializados random)."""
    return np.random.randn(n_patches, dim) * 0.02


def cls_token(dim):
    return np.random.randn(1, dim) * 0.02


def transformer_block(x, W_qkv, W_mlp1, W_mlp2, n_heads=12):
    """Bloque transformer con multi-head attention y MLP."""
    # Pre-norm
    x_norm = (x - x.mean(axis=-1, keepdims=True)) / (
        x.std(axis=-1, keepdims=True) + 1e-6
    )
    # Self-attention (simplificado)
    q = x_norm @ W_qkv[:x.shape[-1]]
    k = x_norm @ W_qkv[x.shape[-1]:2 * x.shape[-1]]
    v = x_norm @ W_qkv[2 * x.shape[-1]:]
    attn = q @ k.T / np.sqrt(q.shape[-1] // n_heads)
    attn = np.exp(attn - attn.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)
    x = x + attn @ v
    # MLP
    x_norm = (x - x.mean(axis=-1, keepdims=True)) / (
        x.std(axis=-1, keepdims=True) + 1e-6
    )
    h = np.maximum(0, x_norm @ W_mlp1)
    x = x + h @ W_mlp2
    return x
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vit
fase: 04
leccion: 14
---

Eres un asistente que ayuda a elegir y aplicar vision
transformers. Recibirás el dataset, el hardware, y la
latencia objetivo. Tu trabajo:

1. Si dataset grande (> 1M imágenes): ViT-L o ViT-H
   preentrenado en ImageNet-21k.
2. Si quieres velocidad: Swin-T o MaxViT-T.
3. Si quieres SOTA: EVA-02 o ConvNeXt-XXL.
4. Si dataset pequeño: usar ViT preentrenado, NO
   entrenar desde cero.
5. Augmentations: RandAugment, MixUp, CutMix.
6. Optimizer: AdamW con lr=1e-4 (fine-tuning) o
   1e-3 (from scratch con warmup).
7. Schedule: cosine decay con 5% warmup.
8. Patch size 16 para 224x224, 14 para 384x384.
9. Evaluar con top-1 / top-5 accuracy sobre ImageNet.
```

## Ejercicios

1. **ViT desde cero**: implementa ViT-T y entrena en
   CIFAR-10.
2. **Patch size**: experimenta con patch size 8 vs 16
   vs 32.
3. **Desafío**: fine-tunea un ViT preentrenado en un
   dataset pequeño y compara con ResNet.

## Lecturas recomendadas

- *An Image is Worth 16x16 Words* — Dosovitskiy et al.,
  2020.
- *Swin Transformer* — Liu et al., 2021.
- *DeiT III* — Touvron et al., 2022.
- *EVA-02* — Fang et al., 2023.
- timm: <https://github.com/huggingface/pytorch-image-models>.

---

> 📚 **Adaptación al español** de la lección "[Vision Transformers (ViT)]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
