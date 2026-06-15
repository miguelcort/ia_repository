# 05 — Transfer learning y fine-tuning

> Reutilizar un modelo preentrenado es la diferencia entre accuracy 0.5 y 0.9 con el mismo dataset. Es la palanca más poderosa del deep learning aplicado.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 04-clasificacion-de-imagenes
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Cargar modelos preentrenados de torchvision o Hugging
  Face.
- Implementar feature extraction y fine-tuning.
- Aplicar discriminative learning rates por capa.
- Diagnosticar cuándo el transfer learning falla y cómo
  ajustarlo.

## El problema

Tienes 1000 imágenes de tu dominio (no ImageNet). Entrenar
ResNet-50 desde cero dará 50-60% accuracy. Usar ResNet-50
preentrenado en ImageNet y *fine-tunear* en tus 1000
imágenes dará 80-90% accuracy. ¿Por qué? Porque las capas
tempranas de ResNet ya aprendieron a detectar bordes,
texturas, y patrones visuales — features universales
útiles para casi cualquier tarea de visión. La lección
cubre cómo aprovechar ese conocimiento.

## El concepto

**¿Por qué funciona?** Las CNNs profundas aprenden una
jerarquía de features. Las capas tempranas detectan bordes
y gradientes. Las intermedias detectan texturas y partes.
Las tardías combinan en conceptos semánticos. Las capas
tempranas son **universales**; las tardías son
**específicas de la tarea**. Por eso, transferir las
primeras capas funciona en casi cualquier dominio.

**Estrategia 1: feature extraction.** Congela todas las
capas convolucionales y entrena solo la cabeza (la última
FC). Útil cuando el dataset es muy pequeño y no quieres
sobreajustar los pesos preentrenados.

**Estrategia 2: fine-tuning.** Descongela todas las
capas (o las últimas K) y entrena con lr bajo. Da más
accuracy pero requiere más datos y cuidado con el learning
rate.

**Estrategia 3: discriminative learning rates.** Aplica
un lr bajo a las capas tempranas y un lr más alto a las
capas tardías. Las capas tempranas solo necesitan ajustes
finos; las tardías necesitan más adaptación al nuevo
dominio.

```python
optimizer = torch.optim.AdamW([
    {"params": model.conv1.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])
```

**Cuándo usar cada estrategia.**

| Dataset | Estrategia |
|---|---|
| < 1k imágenes | Feature extraction (cabeza solamente) |
| 1k - 10k | Fine-tuning con discriminative LR |
| 10k - 100k | Fine-tuning completo con lr bajo |
| > 100k | Fine-tuning + entrenar desde cero opcional |

**Cómo elegir el learning rate.** Típico:

- Solo cabeza: `lr = 1e-3` a `1e-2`.
- Fine-tuning completo: `lr = 1e-4` a `1e-5`.
- Discriminative: `1e-5` (capa 1) a `1e-3` (cabeza).

**Cuando transfer learning falla.**

- **Tu dominio es muy distinto de ImageNet:** las features
  preentrenadas no son útiles. Ejemplo: imágenes
  microscópicas, satélite, médicas. Solución: preentrenar
  en el dominio primero o usar modelos auto-supervisados.
- **Tu tarea tiene clases que no aparecen en ImageNet:**
  difícil. Fine-tuning agresivo y augmentations.
- **Las clases están desbalanceadas:** class weights o
  oversampling.

**Modelos preentrenados comunes.**

- **torchvision:** ResNet, VGG, DenseNet, EfficientNet,
  ConvNeXt, ViT, Swin.
- **timm (Hugging Face):** cientos de modelos, siempre
  actualizados.
- **OpenCLIP:** para zero-shot classification.
- **DINOv2, MAE:** modelos auto-supervisados como
  backbones.

**Trampas.**

- **Olvidar `pretrained=True`:** entrenarás desde cero por
  accidente.
- **No normalizar igual que el preentrenamiento:** cada
  modelo tiene su propia normalización (ImageNet stats es
  el default).
- **Fine-tuning con lr alto:** destruyes los pesos
  preentrenados. Empieza bajo, monitorea train loss.
- **No congelar BatchNorm:** si haces feature extraction,
  congela también los running stats de BatchNorm (o usa
  `model.eval()` durante evaluación).

## Constrúyelo

```python
import numpy as np


def cargar_resnet_pesos_pretrained(modelo, pesos_path, freeze_until="layer3"):
    """Carga pesos preentrenados y congela capas tempranas.
    freeze_until: nombre de la última capa a congelar."""
    pesos = np.load(pesos_path, allow_pickle=True).item()
    modelo.load_state_dict(pesos)
    for name, param in modelo.named_parameters():
        if freeze_until in name:
            break
        param.requires_grad = False
    return modelo


def discriminative_lr(optimizer, base_lr=1e-4, decay=0.5):
    """Aplica lr que decae con la profundidad."""
    for i, group in enumerate(optimizer.param_groups):
        group["lr"] = base_lr * (decay ** (len(optimizer.param_groups) - 1 - i))
    return optimizer


def congelar_capas(modelo, n_capas_a_congelar):
    """Congela las primeras n_capas_a_congelar capas."""
    layers = list(modelo.children())
    for i, layer in enumerate(layers[:n_capas_a_congelar]):
        for param in layer.parameters():
            param.requires_grad = False
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-transfer-learning
fase: 04
leccion: 05
---

Eres un asistente que ayuda a aplicar transfer learning.
Recibirás el dataset (n, número de clases), el dominio, y
el modelo preentrenado disponible. Tu trabajo:

1. Si n < 1k: feature extraction (cabeza sola), lr=1e-3.
2. Si 1k < n < 10k: fine-tuning con discriminative LR
   (1e-5 a 1e-3).
3. Si n > 10k: fine-tuning completo, lr=1e-4.
4. Si el dominio es muy distinto (médico, satélite):
   preentrenar con auto-supervisado primero.
5. Congelar BatchNorm si haces feature extraction.
6. Normalizar inputs igual que el preentrenamiento.
7. Monitorear train loss: si sube al inicio, lr muy alto.
8. Recomienda timm para la colección más amplia de
   modelos.
```

## Ejercicios

1. **Feature extraction**: implementa el flujo
   feature extraction con ResNet-50 y Entrenamiento solo de
   la cabeza.
2. **Discriminative LR**: implementa un optimizer con
   learning rates por capa.
3. **Desafío**: aplica transfer learning a un dataset de
   tu elección y compara con entrenar desde cero.

## Lecturas recomendadas

- *How transferable are features in deep neural networks?* —
  Yosinski et al., 2014.
- *Deep Learning* — Goodfellow et al. (sec. sobre
  transfer learning).
- torchvision models: <https://pytorch.org/vision/stable/models.html>.
- timm: <https://github.com/huggingface/pytorch-image-models>.

---

> 📚 **Adaptación al español** de la lección "[Transfer Learning and Fine-Tuning]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
