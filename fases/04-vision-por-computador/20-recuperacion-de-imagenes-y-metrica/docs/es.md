# Recuperación de imágenes y metric learning

> Buscar las N imagenes mas similares a una query en millones. Encoder (CLIP/DINOv2) + indice ANN (FAISS/Milvus) + metricas (Recall@K, mAP).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18-clip-vocabulario-abierto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Construir un indice de embeddings.
- Buscar top-k por similitud coseno.
- Calcular Recall@K, Precision@K, mAP.
- Diagnosticar re-ranking y FAISS.

## Constrúyelo

```python
def construir_indice(embeddings):
    return np.array([e / np.linalg.norm(e) for e in embeddings])


def buscar_por_similitud(query, indice, top_k=5):
    q = query / np.linalg.norm(query)
    sims = indice @ q
    idx = np.argsort(sims)[::-1][:top_k]
    return idx, sims[idx]
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

1. <1M: FAISS HNSW + DINOv2/CLIP.
2. >10M: IVF-PQ, cuantizacion.
3. Re-rank: cross-encoder o k-reciprocal.
4. CLIP dual encoder para multi-modal.
5. mAP@100, Recall@10.
```

## Ejercicios

1. **FAISS**: instalar y benchmarkear HNSW vs IVF en 100K
   embeddings.
2. **Triplet loss**: entrenar encoder con margin loss.
3. **Desafio**: construir pipeline completo con FAISS +
   CLIP y benchmarkear mAP en ROxford.

## Lecturas recomendadas

- "FaceNet" (Schroff et al., 2015) — triplet loss
- "ArcFace" (Deng et al., 2019) — angular margin
- FAISS: <https://github.com/facebookresearch/faiss>

---

> 📚 **Adaptación al español** de la lección "[Image Retrieval & Metric Learning]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).