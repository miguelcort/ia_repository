# NLP multilingue

> Un modelo para muchos idiomas. mBERT (104), XLM-R (100), mT5 (101), NLLB (200), BLOOM (46). Tokenizacion SentencePiece compartida. Zero-shot cross-lingual transfer. BETO y MarIA para espanol fine-tune.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17-chatbots-de-reglas-a-neuronal
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar deteccion de idioma basica.
- Implementar alineacion de palabras (LCS).
- Diagnosticar modelos multilingues.
- Diagnosticar monolingual vs multilingual fine-tune.

## Constrúyelo

```python
def align_palabras(src_tokens, tgt_tokens):
    # LCS
    m, n = len(src_tokens), len(tgt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_tokens[i - 1] == tgt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # Backtrack
    pares = []
    i, j = m, n
    while i > 0 and j > 0:
        if src_tokens[i - 1] == tgt_tokens[j - 1]:
            pares.append((i - 1, j - 1))
            i -= 1; j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return pares[::-1]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-multilingual
fase: 05
leccion: 18
---

1. Multi-idioma: XLM-R, mT5, mBART.
2. Espanol: BETO, RoBERTa-es, MarIA.
3. MT: NLLB-200, OPUS-MT.
4. Zero-shot: mBERT, XLM-R + transfer.
5. SentencePiece BPE compartido.
```

## Ejercicios

1. **fastText langdetect**: usar fasttext library para
   deteccion de idioma precisa.
2. **Align con FastAlign**: implementar IBM Model 1/2 para
   word alignment.
3. **Desafio**: fine-tunear XLM-R en NER espanol usando
   CoNLL-2002 subset, alcanzar F1 > 0.85.

## Lecturas recomendadas

- "Multilingual BERT" (Devlin et al., 2019) — mBERT
- "XLM-RoBERTa" (Conneau et al., 2020)
- "SentencePiece" (Kudo & Richardson, 2018)

---

> 📚 **Adaptación al español** de la lección "[Multilingual NLP]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).