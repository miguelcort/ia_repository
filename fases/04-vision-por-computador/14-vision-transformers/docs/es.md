# Vision Transformers (ViT)

> Transformer toma el control de vision: divide la imagen en patches, los proyecta como tokens, y aplica self-attention. Sin convoluciones, global receptive field desde la primera capa.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar patch embedding.
- Implementar positional encoding senos/cosenos.
- Implementar self-attention.
- Construir pipeline de clasificacion con ViT.

## Constrúyelo

```python
def self_attention_simple(x, dim_head=64):
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    scores = Q @ K.T / np.sqrt(dim_head)
    pesos = np.exp(scores - scores.max(axis=-1, keepdims=True))
    pesos /= pesos.sum(axis=-1, keepdims=True)
    return pesos @ V
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vit-elegir
fase: 04
leccion: 14
---

1. Max accuracy: ViT-L, EVA-02, JFT preentrenado.
2. Pocos datos: DeiT, ConvNeXt.
3. Deteccion: Swin.
4. SSL: MAE, DINOv2.
5. Mobile: MobileViT.
```

## Ejercicios

1. **Multi-head attention**: implementa 8 cabezas en
   paralelo.
2. **MLP block**: implementa el bloque FFN del Transformer.
3. **Desafio**: implementar ViT completo y entrenar en
   CIFAR-10.

## Lecturas recomendadas

- "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- "Swin Transformer" (Liu et al., 2021)
- "DeiT" (Touvron et al., 2021)
- timm: <https://github.com/huggingface/pytorch-image-models>

---

> 📚 **Adaptación al español** de la lección "[Vision Transformers]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).