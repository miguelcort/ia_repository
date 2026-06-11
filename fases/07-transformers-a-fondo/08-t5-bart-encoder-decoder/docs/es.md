# T5 y BART encoder-decoder

> T5: text-to-text transfer con span corruption (mask N tokens consecutivos → predecir span). BART: denoising autoencoder con 5 noises (token masking, deletion, text infilling, sentence permutation, document rotation). Encoder-decoder con cross-attention. Útiles para traducción, summarization, QA generativo, regression.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar span corruption estilo T5.
- Implementar shift-right para decoder input.
- Implementar BART noise functions.
- Diagnosticar encoder-decoder vs decoder-only.

## Constrúyelo

```python
def span_corrupt(token_ids, mask_id, span_length=3, ratio=0.15, seed=0):
    n = len(token_ids)
    n_mask = max(1, int(n * ratio))
    starts = sorted(rng.choice(n, size=min(n_mask, n), replace=False))
    targets = np.full(n, -100)
    input_ids = list(token_ids)
    for s in starts:
        e = min(n, s + int(rng.integers(1, span_length + 1)))
        targets[s:e] = token_ids[s:e]
        for i in range(s, e):
            input_ids[i] = mask_id
    return np.array(input_ids), targets
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-encoder-decoder
fase: 07
leccion: 08
---

1. T5: text-to-text, span corruption.
2. BART: 5 noises (mask, delete, infill, permute, rotate).
3. Cross-attn: Q=decoder, K=V=encoder.
4. Decoder-only para chat/code.
5. Encoder-decoder para seq2seq.
```

## Ejercicios

1. **Comparar**: entrenar T5-small vs decoder-only
   en summarization. Quality y costo.
2. **BART noises**: pre-train con cada noise
   individualmente, comparar downstream.
3. **Desafio**: implementar Flan-T5 (instruction
   tuning) y medir zero-shot.

## Lecturas recomendadas

- "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2020)
- "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation" (Lewis et al., 2020)
- "UL2: Unifying Language Learning Paradigms" (Tay et al., 2023)

---

> 📚 **Adaptación al español** de la lección "[T5 BART Encoder Decoder]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).