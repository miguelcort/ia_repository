# BERT masked language modeling

> BERT: encoder bidireccional pre-entrenado con MLM (mask 15% tokens: 80% [MASK] + 10% random + 10% unchanged) + NSP. Bidireccional = cada token ve contexto completo. Tareas downstream: classification ([CLS]), NER, QA, similarity. Variantes: RoBERTa (dynamic mask), ALBERT (SOP), ELECTRA (RTD), DeBERTa (disentangled attention).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar mask tokens con distribución 80/10/10.
- Implementar cross-entropy MLM loss.
- Diagnosticar NSP vs SOP.
- Aplicar BERT a tareas downstream.

## Constrúyelo

```python
def mask_tokens(token_ids, mask_id, vocab_size, mlm_prob=0.15, seed=0):
    labels = np.full(len(token_ids), -100)
    prob_matrix = rng.uniform(size=len(token_ids)) < mlm_prob
    prob_matrix &= ~np.isin(token_ids, [0, 1])  # no [CLS], [SEP]
    indices = np.where(prob_matrix)[0]
    labels[indices] = token_ids[indices]
    # 80% MASK, 10% random, 10% unchanged
    ...
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-bert-mlm
fase: 07
leccion: 06
---

1. MLM: 80/10/10 distribution.
2. NSP: binario is_next. SOP: predice orden.
3. Fine-tuning: lr 2e-5, 2-4 epochs.
4. BERT bidireccional, GPT unidireccional.
5. RoBERTa, ALBERT, ELECTRA son improvements.
```

## Ejercicios

1. **Dynamic masking**: implementar mask que cambia
   por epoch.
2. **Span corruption**: predecir 15% de tokens
   consecutivos (T5-style).
3. **Desafio**: implementar ELECTRA-style replaced
   token detection.

## Lecturas recomendadas

- "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
- "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)
- "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" (Clark et al., 2020)

---

> 📚 **Adaptación al español** de la lección "[BERT Masked Language Modeling]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).