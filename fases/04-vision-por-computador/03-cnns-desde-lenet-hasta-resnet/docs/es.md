# CNNs: de LeNet a ResNet

> Veinte anos de evolucion: de LeNet (2 capas, MNIST) a ResNet-152 y EfficientNet. La historia de las CNN es la historia de encontrar mejores formas de apilar convs sin que la red colapse.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-convoluciones-desde-cero
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Modelar bloques Conv+BN+ReLU.
- Calcular receptive field y parametros.
- Entender la evolucion LeNet -> AlexNet -> VGG -> ResNet.

## Constrúyelo

```python
def cuente_params_conv(in_c, out_c, k=3):
    return in_c * k * k * out_c + out_c
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-cnn-arquitectura
fase: 04
leccion: 03
---

1. Pequeno: ResNet-18 / EfficientNet-B0.
2. Grande: ResNet-50/101, ConvNeXt.
3. Mobile: MobileNetV3.
4. Skip connections siempre (ResNet-style).
5. RF > tamano del objeto.
```

## Ejercicios

1. **Inception module**: implementa la paralela 1x1, 3x3,
   5x5, pool.
2. **ResNet-18 completo**: implementa arquitectura con
   basic blocks.
3. **Desafio**: entrena ResNet-18 en CIFAR-10 con data
   augmentation y reporta accuracy.

## Lecturas recomendadas

- "Deep Residual Learning" (He et al., 2015)
- "EfficientNet" (Tan & Le, 2019)
- "A Survey of Deep Learning Techniques for Neural Network
  Architectures" (Bender, 2020)

---

> 📚 **Adaptación al español** de la lección "[CNNs: LeNet to ResNet]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).