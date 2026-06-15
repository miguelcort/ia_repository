# 04 — Clasificación de imágenes

> El problema base de visión: dada una imagen, asigna una etiqueta. ResNet preentrenado en ImageNet resuelve el 90% de los casos industriales.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Entrenar un clasificador de imágenes desde cero con
  PyTorch.
- Aplicar transfer learning desde ImageNet a un dominio
  nuevo.
- Diagnosticar errores comunes en clasificación de
  imágenes.
- Implementar augmentations y evaluar con top-1 / top-5
  accuracy.

## El problema

Tienes 10k fotos de productos en 50 categorías y quieres
clasificarlas automáticamente. Entrenar una CNN desde cero
con 10k imágenes dará ~50% accuracy. Usar ResNet-50
preentrenado en ImageNet y *fine-tunear* en tus 10k fotos
dará ~85% accuracy en una tarde. La lección cubre el flujo
completo: data loading, augmentations, transfer learning,
entrenamiento, evaluación.

## El concepto

**Dataset pipeline.** En producción, los datos viven en S3/GCS
o disco local. PyTorch `Dataset` y `DataLoader` los cargan
bajo demanda con `num_workers` paralelos. Cada batch se
augmenta al vuelo.

**Augmentations críticas en visión.**

- **RandomResizedCrop:** toma un crop aleatorio y lo
  redimensiona a 224x224. Simula variabilidad de escala.
- **RandomHorizontalFlip:** con probabilidad 0.5. Asume que
  la categoría no cambia con el flip.
- **ColorJitter:** cambia brillo, contraste, saturación,
  tono. Simula diferentes condiciones de iluminación.
- **RandAugment / AutoAugment:** estrategias de
  augmentation buscadas automáticamente.
- **MixUp / CutMix:** combinan imágenes y etiquetas. Mejora
  calibración y robustez.

**Transfer learning.** El flujo estándar:

1. Cargar ResNet-50 preentrenado en ImageNet.
2. Reemplazar la última capa FC con `nn.Linear(in_features,
   num_clases_nuevas)`.
3. **Estrategia 1: feature extraction.** Congelar todos los
   pesos excepto la nueva FC. Entrenar pocos epochs.
4. **Estrategia 2: fine-tuning.** Descongelar todo (o las
   últimas N capas) y entrenar con lr bajo (1e-4).
5. **Discriminative learning rates:** lr bajo para capas
   tempranas, lr más alto para capas tardías.

**Optimización.** AdamW con lr=1e-4 (fine-tuning completo)
o lr=1e-3 (solo la cabeza), cosine decay, ~10-30 epochs,
batch size 32-128.

**Evaluación.** Top-1 y top-5 accuracy sobre test set. Si
el dataset está desbalanceado, también precision y recall
ponderadas. Matriz de confusión para entender qué clases
confunde el modelo.

**Errores comunes.**

- **Olvidar `model.eval()` antes de validar:** dropout y
  BatchNorm se comportan distinto en train/val.
- **No normalizar igual que preentrenamiento:** si preentrenaste
  con ImageNet stats, usarlas también en fine-tuning.
- **Augmentation demasiado agresiva:** imágenes irreconocibles.
  Validar visualmente.
- **Learning rate muy alto en fine-tuning:** destruyes los
  pesos preentrenados. Usa lr=1e-4 o menor.

**Cuándo entrenar desde cero vs transfer learning.**

| Situación | Recomendación |
|---|---|
| Tu dominio es similar a ImageNet | Transfer learning |
| Tu dominio es muy distinto (satélite, histopatología) | Transfer con fine-tuning fuerte |
| Dataset enorme (millones) | Posible desde cero |
| Dataset pequeño (< 10k) | Transfer learning obligatorio |
| Necesitas máxima accuracy | Ensemble de modelos preentrenados |

## Constrúyelo

```python
import numpy as np


def top_k_accuracy(y_true, scores, k=5):
    """Top-k accuracy: la clase correcta está entre las k
    predicciones con mayor score."""
    top_k = np.argsort(scores, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in top_k[i] for i in range(len(y_true))]))


def confusion_matrix(y_true, y_pred, n_clases):
    M = np.zeros((n_clases, n_clases), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


def per_class_accuracy(M):
    """Accuracy por clase: diagonal / suma de fila."""
    return np.diag(M) / np.maximum(M.sum(axis=1), 1)


def augmentar_imagen(im, modo="train"):
    """Augmentations mínimas sin torchvision."""
    if modo == "train" and np.random.rand() > 0.5:
        im = im[:, ::-1].copy()  # flip horizontal
    return im


def class_weights(y, n_clases):
    """Pesos inversos a la frecuencia para clases desbalanceadas."""
    counts = np.bincount(y, minlength=n_clases)
    return (len(y) / (n_clases * counts)).astype(float)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-clasificacion-imagen
fase: 04
leccion: 04
---

Eres un asistente que ayuda a entrenar un clasificador de
imágenes. Recibirás el tamaño del dataset, el número de
clases, y el hardware. Tu trabajo:

1. Si n < 10k: usar transfer learning con ResNet-50 o
   ConvNeXt preentrenado en ImageNet.
2. Si n > 100k: considerar entrenar desde cero o
   fine-tuning fuerte.
3. Augmentations: RandomResizedCrop + HorizontalFlip +
   ColorJitter.
4. Optimizer: AdamW con lr=1e-4 (fine-tuning) o 1e-3
   (solo cabeza).
5. Schedule: cosine decay sobre 10-30 epochs.
6. Batch size: 32-128 según GPU.
7. Evaluación: top-1 y top-5 accuracy, matriz de
   confusión, precision/recall por clase.
8. Recomienda monitorear loss y accuracy con TensorBoard.
```

## Ejercicios

1. **Transfer learning**: fine-tunea ResNet-18 en CIFAR-10
   y compara con entrenar desde cero.
2. **Augmentations**: visualiza el efecto de RandAugment en
   un batch de imágenes.
3. **Desafío**: implementa un ensemble de 3 ResNets con
   diferentes semillas y promedia sus predicciones.

## Lecturas recomendadas

- *Deep Learning* — Goodfellow et al. (cap. 9).
- *Hands-On Machine Learning* — Géron (2nd ed., cap. 14).
- torchvision: <https://pytorch.org/vision/stable/index.html>.
- timm (PyTorch Image Models): <https://github.com/huggingface/pytorch-image-models>.

---

> 📚 **Adaptación al español** de la lección "[Image Classification]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
