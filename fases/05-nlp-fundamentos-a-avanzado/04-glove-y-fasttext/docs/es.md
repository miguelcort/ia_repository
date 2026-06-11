# GloVe y fastText

> GloVe: factorizacion de matriz de co-ocurrencia con bias. fastText: extension de word2vec con subword n-grams (maneja OOV, multilingue). Los dos son estandar en produccion donde BERT es overkill.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-embeddings-de-palabras-word2vec
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Calcular matriz de co-ocurrencia.
- Implementar GloVe loss simplificado.
- Generar subword n-grams (fastText).
- Diagnosticar GloVe vs fastText vs BERT.

## Constrúyelo

```python
def fasttext_subword(word, n_min=3, n_max=6):
    padded = "<" + word + ">"
    subs = [padded]
    for n in range(n_min, min(n_max + 1, len(padded))):
        for i in range(len(padded) - n + 1):
            subs.append(padded[i:i + n])
    return subs
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-glove-fasttext
fase: 05
leccion: 04
---

1. Espanol baseline: fastText wiki.es, SBW.
2. Clasif pocos datos: fastText trainable.
3. Contexto: BETO, RoBERTa-es, XLM-R.
4. OOV: fastText subwords.
5. Para empezar: fastText preentrenado.
```

## Ejercicios

1. **GloVe completo**: implementar el algoritmo de
   factorizacion.
2. **fastText trainable**: usar fasttext.train_supervised
   en un dataset de clasificacion.
3. **Desafio**: entrenar fastText en 100K tweets en
   espanol para clasificar sentiment, alcanzar F1 > 0.85.

## Lecturas recomendadas

- "GloVe" (Pennington et al., 2014)
- "Enriching Word Vectors with Subword Information"
  (Bojanowski et al., 2017) — fastText
- fastText: <https://fasttext.cc/>

---

> 📚 **Adaptación al español** de la lección "[GloVe and fastText]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).