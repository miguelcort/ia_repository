# Transfer learning

> Entrenar de cero con pocos datos es tirar el tiempo. Un modelo preentrenado en ImageNet ya sabe bordes, texturas y formas; tu trabajo es adaptar la cabeza a tu tarea.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 04-clasificacion-de-imagenes
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Entender feature extraction vs fine-tuning.
- Aplicar learning rate diferencial.
- Diagnosticar cuando descongelar capas.

## Constrúyelo

```python
def estrategia_lr_diferencial(base_lr, ratio_backbone=0.1):
    return {
        "backbone": base_lr * ratio_backbone,
        "cabeza": base_lr,
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-transfer-strategy
fase: 04
leccion: 05
---

1. <1K/clase: feature extraction.
2. 1K-10K: fine-tune parcial.
3. 10K-100K: fine-tune completo.
4. >100K: de cero o fine-tune.
5. AdamW lr=1e-3 cabeza, 1e-4 backbone.
```

## Ejercicios

1. **Discriminative LR**: implementa lr_layerwise_decay.
2. **Gradual unfreezing**: descongela capas progresivamente.
3. **Desafio**: fine-tune ResNet-18 en un dataset custom
   (e.g. tu propia coleccion de imagenes).

## Lecturas recomendadas

- "How transferable are features in deep neural networks?"
  (Yosinski et al., 2014)
- "A Survey on Transfer Learning" (Pan & Yang, 2010)
- timm: <https://github.com/huggingface/pytorch-image-models>

---

> 📚 **Adaptación al español** de la lección "[Transfer Learning]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).