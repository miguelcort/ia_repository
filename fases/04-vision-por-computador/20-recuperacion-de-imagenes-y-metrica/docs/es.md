# 20 — Recuperación de imágenes y metric learning

> Buscar imágenes similares a una imagen de query, o mapear imágenes a un espacio donde imágenes similares están cerca.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17-vision-auto-supervisada,
                  18-clip-vocabulario-abierto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar triplet loss y ArcFace.
- Calcular Recall@K sobre un dataset de retrieval.
- Usar embeddings preentrenados (DINOv2, CLIP) para
  retrieval.
- Diagnosticar y elegir la métrica de distancia correcta.

## El problema

Tienes 10M de imágenes de productos y el usuario sube una
foto. ¿Cuáles son las 5 imágenes más similares? Necesitas
representar cada imagen como un vector (embedding) y
buscar los K vecinos más cercanos en ese espacio. La
lección cubre cómo entrenar embeddings (triplet loss,
ArcFace) y cómo indexarlos (FAISS, ScaNN) para búsqueda
rápida.

## El concepto

**Triplet loss.** Para un anchor `a`, un positivo `p` (de la
misma clase) y un negativo `n` (de otra clase):

```text
L = max(0, d(a, p) - d(a, n) + margin)
```

El modelo aprende a colocar el anchor más cerca del
positivo que del negativo por al menos `margin`. Requiere
**triplet mining**: hard negatives (difíciles) son los más
útiles.

**ArcFace (Deng et al., 2019).** Función de pérdida para
clasificación con margen angular. Aprende embeddings
discriminativos al añadir un margen angular `m` entre
clases. Mejora sobre softmax cross-entropy estándar para
face recognition y retrieval.

**Circle loss, SupCon, InfoNCE.** Variantes más recientes
del contrastive loss con diferentes propiedades.

**Métricas de evaluación.**

- **Recall@K:** de las queries, ¿en qué porcentaje el
  positivo aparece en los top-K?
- **mAP (mean Average Precision):** promedio de la
  precision a diferentes recalls.
- **NDCG@K:** considera el ranking, no solo presencia.

**Embeddings preentrenados.** Para no entrenar desde cero:

- **DINOv2:** state of the art en features genéricas.
  Linear probe SOTA en clasificación, segmentation, depth.
- **CLIP:** features multimodales (imagen + texto).
- **EVA-02:** ViT a gran escala preentrenado.
- **DINOv2 + retrieval:** ideal para producto visual
  search.

**Indexing para búsqueda rápida.**

- **FAISS (Facebook AI Similarity Search):** indexa millones
  de vectores con búsqueda sub-lineal. Soporta GPU.
- **ScaNN (Google):** state of the art en
  approximate nearest neighbor.
- **Annoy (Spotify):** basado en árboles aleatorios,
  simple.

**Trampas.**

- **Triplet mining pobre:** triplet loss con negativos
  fáciles no aprende nada. Usar hard negative mining.
- **Sin normalización L2:** cosine similarity sin
  normalización no es comparable.
- **Index muy grande:** 10M de vectores float32 son 120
  GB. Quantizar a int8 reduce 4x.

## Constrúyelo

```python
import numpy as np


def triplet_loss(anchor, positive, negative, margin=0.2):
    """Triplet loss con distancia euclidiana."""
    d_ap = np.sqrt(np.sum((anchor - positive) ** 2, axis=-1))
    d_an = np.sqrt(np.sum((anchor - negative) ** 2, axis=-1))
    return float(np.mean(np.maximum(0, d_ap - d_an + margin)))


def arcface_loss(embeddings, labels, num_classes, m=0.5, s=30):
    """ArcFace: añade margen angular m en coseno."""
    # Normalizar
    emb = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    W = np.random.randn(embeddings.shape[-1], num_classes)
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    cos_theta = emb @ W  # (batch, num_classes)
    # Para la clase correcta, añadir margen
    theta = np.arccos(np.clip(cos_theta, -1 + 1e-7, 1 - 1e-7))
    target_theta = theta[labels]
    cos_target = np.cos(target_theta + m)
    # Reemplazar el coseno de la clase correcta
    cos_theta[np.arange(len(labels)), labels] = cos_target - cos_theta[
        np.arange(len(labels)), labels
    ]
    logits = s * cos_theta
    # Cross-entropy
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return float(-np.mean(
        np.log(exp_logits[np.arange(len(labels)), labels] /
               exp_logits.sum(axis=-1))
    ))


def recall_at_k(query_emb, db_embs, query_labels, db_labels, k=10):
    """Recall@K: el positivo aparece en los top-K?"""
    similarities = query_emb @ db_embs.T
    n_hits = 0
    for i, sim in enumerate(similarities):
        top_k = np.argsort(sim)[-k:]
        if query_labels[i] in db_labels[top_k]:
            n_hits += 1
    return n_hits / len(query_emb)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-retrieval
fase: 04
leccion: 20
---

Eres un asistente que ayuda a construir un sistema de
recuperación de imágenes. Recibirás el dataset, la
latencia y la precisión objetivo. Tu trabajo:

1. Para SOTA: usar DINOv2 como encoder de features.
2. Si quieres multimodal: usar CLIP.
3. Si necesitas training: triplet loss o ArcFace.
4. Indexar con FAISS (HNSW o IVF para > 1M vectores).
5. Quantizar embeddings a int8 si memoria es limitante.
6. Métrica: Recall@1, Recall@10, mAP.
7. Para re-ranking: usar un cross-encoder después de
   la búsqueda inicial.
8. Latencia: < 50 ms para top-10 sobre 1M vectores
   con FAISS-GPU.
```

## Ejercicios

1. **Triplet loss**: implementa y entrena embeddings con
   triplet loss.
2. **ArcFace**: implementa ArcFace y compara con
   softmax estándar.
3. **Desafío**: indexa 100k imágenes con FAISS y
   mide latencia P50.

## Lecturas recomendadas

- *FaceNet: A Unified Embedding for Face Recognition* —
  Schroff et al., 2015.
- *ArcFace: Additive Angular Margin Loss for Deep Face
  Recognition* — Deng et al., 2019.
- *DINOv2* — Oquab et al., 2023.
- FAISS: <https://github.com/facebookresearch/faiss>.

---

> 📚 **Adaptación al español** de la lección "[Image Retrieval and Metric Learning]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
