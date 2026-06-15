# 17 — Visión auto-supervisada: SimCLR, DINO, MAE

> Cuando no hay etiquetas, los métodos auto-supervisados aprenden representaciones visuales útiles preentrenando en datos sin anotar.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-cnns-desde-lenet-hasta-resnet
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar SimCLR (contrastive) y MAE (masked
  autoencoding) desde cero.
- Diagnosticar representaciones aprendidas con linear
  probing.
- Conocer DINO, MoCo, MAE, DINOv2 y cuándo usar cada
  uno.
- Aplicar transfer learning desde modelos auto-supervisados.

## El problema

Etiquetar imágenes es caro. ImageNet tiene 1.2M imágenes
etiquetadas (14M con labels). Pero hay miles de millones
de imágenes sin etiquetar en internet. Los métodos
auto-supervisados aprenden **representaciones útiles** de
imágenes sin etiquetas, preentrenando en datasets masivos.
Luego se aplica transfer learning a tareas downstream.
DINOv2 (Meta, 2023) es SOTA y se entrena con 142M de
imágenes sin etiquetar.

## El concepto

**Contrastive learning (SimCLR, Chen et al., 2020).** Dadas
dos augmentations de la misma imagen (par positivo),
entrena al modelo a producir embeddings similares.
Augmentations de imágenes distintas son pares negativos.
Loss InfoNCE:

```text
L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
```

donde `τ` es la temperatura. Requiere batches grandes
(4096+) para tener suficientes negativos.

**MoCo (He et al., 2020).** Mantiene un queue de
embeddings negativos y un encoder momentum (actualizado
como exponential moving average del encoder principal).
Permite batches pequeños.

**DINO (Caron et al., 2021).** Self-distillation: un
estudiante y un teacher (EMA) procesan dos views
diferentes. El estudiante predice la distribución del
teacher. No requiere negativos.

**Masked autoencoding (MAE, He et al., 2022).** Enmascara
el 75% de los patches de la imagen. Un encoder
(generalmente ViT) procesa solo los patches visibles.
Un decoder pequeño reconstruye los patches enmascarados.
Loss: MSE en píxeles (o en features de un teacher ViT,
llamado MAE-Lite).

**DINOv2 (Oquab et al., 2023).** SOTA en visión auto-
supervisada. Combina DINO + iBOT (masked image
modeling). Preentrenado en 142M de imágenes. Produce
features que son SOTA en clasificación, segmentación, y
depth estimation con linear probing o fine-tuning.

**Linear probing.** Métrica estándar para evaluar
representaciones: congela el encoder preentrenado y
entrena solo un clasificador linear encima. Si la
accuracy es alta, las features son útiles.

**Cuándo usar cada método.**

| Método | Cuándo |
|---|---|
| **CLIP** | Cuando tienes pares imagen-texto |
| **DINOv2** | Cuando quieres SOTA features genéricas |
| **MAE** | Cuando quieres entrenar ViT preentrenado |
| **SimCLR** | Baseline, simple de implementar |
| **MoCo** | Si batch size es limitado |

**Trampas.**

- **Augmentations mal elegidas:** SimCLR es muy
  sensible. Color jitter fuerte + crop es esencial.
- **Batches pequeños:** contrastive loss necesita
  muchos negativos. Usa MoCo o DINO si batch es
  limitado.
- **Linear probe muy estrecho:** no captura la
  calidad. Usa k-NN o fine-tuning supervisado.

## Constrúyelo

```python
import numpy as np


def nt_xent_loss(z_i, z_j, tau=0.5):
    """InfoNCE loss (SimCLR). z_i, z_j: (batch, dim)."""
    batch = z_i.shape[0]
    z = np.concatenate([z_i, z_j], axis=0)  # (2*batch, dim)
    sim = z @ z.T / tau  # (2*batch, 2*batch)
    # Máscaras: positivos en (i, i+batch) y (i+batch, i)
    pos_mask = np.zeros((2 * batch, 2 * batch))
    for i in range(batch):
        pos_mask[i, i + batch] = 1
        pos_mask[i + batch, i] = 1
    # Negativos: todos los otros
    neg_mask = 1 - np.eye(2 * batch) - pos_mask
    # Estables softmax
    sim_max = sim.max(axis=1, keepdims=True)
    exp_sim = np.exp(sim - sim_max)
    exp_sim_neg = exp_sim * neg_mask
    denom = exp_sim_neg.sum(axis=1, keepdims=True) + exp_sim[
        np.arange(2 * batch), np.concatenate(
            [np.arange(batch, 2 * batch), np.arange(batch)]
        )
    ].reshape(-1, 1)
    log_prob = sim - sim_max - np.log(denom)
    loss = -np.sum(pos_mask * log_prob) / (2 * batch)
    return float(loss)


def random_mask(patches, mask_ratio=0.75):
    """MAE: enmascara aleatoriamente el 75% de los patches."""
    n = len(patches)
    n_mask = int(n * mask_ratio)
    mask = np.zeros(n, dtype=bool)
    mask[np.random.choice(n, n_mask, replace=False)] = True
    return mask


def mae_reconstruction_loss(pred, target, mask):
    """MSE solo en patches enmascarados."""
    return float(np.mean((pred[mask] - target[mask]) ** 2))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-self-supervised
fase: 04
leccion: 17
---

Eres un asistente que ayuda a aplicar visión auto-
supervisada. Recibirás el dataset, el hardware, y la
tarea downstream. Tu trabajo:

1. Si quieres SOTA features: usar DINOv2 preentrenado
   de Meta.
2. Si quieres preentrenar ViT: MAE.
3. Si quieres entrenar CNN: SimCLR o MoCo.
4. Linear probing: si accuracy > 70%, las features
   son útiles.
5. Fine-tuning siempre supera linear probing pero
   requiere más datos.
6. Augmentations fuertes: random crop, color jitter,
   Gaussian blur.
7. Batch size: 256-4096 para contrastive.
8. Temperature τ=0.5 para SimCLR; ajustar si la
   accuracy es baja.
9. Evaluar con linear probe + k-NN.
```

## Ejercicios

1. **SimCLR**: implementa y entrena en CIFAR-10 con
   SimCLR.
2. **Linear probe**: evalúa las features congeladas con
   un linear classifier.
3. **Desafío**: preentrena un ViT-T con MAE en un
   dataset pequeño y compara linear probe vs from
   scratch.

## Lecturas recomendadas

- *A Simple Framework for Contrastive Learning of Visual
  Representations (SimCLR)* — Chen et al., 2020.
- *Momentum Contrast (MoCo)* — He et al., 2020.
- *Emerging Properties in Self-Supervised Vision Transformers
  (DINO)* — Caron et al., 2021.
- *Masked Autoencoders Are Scalable Vision Learners
  (MAE)* — He et al., 2022.
- *DINOv2* — Oquab et al., 2023.

---

> 📚 **Adaptación al español** de la lección "[Self-Supervised Vision]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
