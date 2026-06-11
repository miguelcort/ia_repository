# Modelado de temas

> Descubrir temas abstractos en una coleccion. LDA (probabilistico, base), NMF (rapido), BERTopic (SOTA con semantica). Aplicaciones: exploracion de corpus, busqueda, organizacion.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14-recuperacion-de-informacion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar LDA con collapsed Gibbs sampling.
- Obtener top-N palabras por topic.
- Evaluar con coherence.
- Diagnosticar LDA vs BERTopic vs NMF.

## Constrúyelo

```python
def lda_gibbs_step(docs, k_topics, n_iter=20, semilla=0):
    # ... collapsed Gibbs sampling
    return theta, phi, vocab
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-topic-modeling
fase: 05
leccion: 15
---

1. Corpus largo: LDA (gensim, sklearn).
2. Short text / multilingual: BERTopic + SBERT.
3. Rapido: NMF sobre TF-IDF.
4. SOTA: BERTopic + UMAP + HDBSCAN.
5. Eval: coherence, topic diversity.
```

## Ejercicios

1. **LDA completo**: implementar LDA con variational
   inference.
2. **BERTopic**: instalar y aplicar a un dataset de
   tweets.
3. **Desafio**: comparar LDA, NMF y BERTopic en un
   corpus de noticias en espanol, reportar coherence y
   diversity.

## Lecturas recomendadas

- "Latent Dirichlet Allocation" (Blei et al., 2003)
- "BERTopic" (Grootendorst, 2022)
- gensim LDA: <https://radimrehurek.com/gensim/models/ldamodel.html>

---

> 📚 **Adaptación al español** de la lección "[Topic Modeling]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).