# ColPali vision native RAG

> ColPali (Faysse 2024): vision-native RAG con document pages como imagenes directas al VLM + PaliGemma 3B + ColBERT-style late interaction (encode query en tokens + encode doc en tokens + sim matrix query x doc + MaxSim max over doc para cada query + sum max sims) + +SOTA document retrieval 2024-25. Variants: ColPali 3B (PaliGemma seminal), ColQwen2 (Qwen2-VL +multilingual +SOTA), SmolVLM (HuggingFace +efficient +SOTA +open), ColSmolVLM (+efficient +SOTA). +Insights: -OCR, -Layout analysis, -Modular, +End-to-end, +Late interaction, +Token-level. Frameworks: colpali, byaldi, transformers, vLLM, qwen-vl. +Production: ColPali + ColQwen2 + SmolVLM SOTA. +Use cases: PDF retrieval, document Q&A, visual RAG, multilingual. Trade-offs: vision-native + SOTA, traditional RAG + modular. Hoy: SOTA 2024-25 standard. 2025: +Native + reasoning + scientific + multilingual.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/02, 12/22
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar colbert_late_interaction con MaxSim.
- Implementar encode_page_pali_gemma y encode_query.
- Implementar colpali_retrieve top-k.
- Implementar colpali_rag retrieve + generate.
- Diagnosticar ColPali vs traditional RAG.

## Constrúyelo

```python
def colbert_late_interaction(doc_tokens, query_tokens):
    sim = query_tokens @ doc_tokens.T
    max_sim = sim.max(axis=1)
    return float(max_sim.sum())
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: colpali
fase: 12
leccion: 23
---

1. Vision-native RAG.
2. PaliGemma + ColBERT.
3. MaxSim + sum.
4. -OCR -Modular.
5. +SOTA 2024-25.
```

## Ejercicios

1. **ColPali**: usar ColPali
   con byaldi.
2. **ColQwen2**: probar
   ColQwen2 multilingual.
3. **Desafio**: ColPali
   para PDF retrieval.

## Lecturas recomendadas

- "ColPali: Efficient Document Retrieval with Vision Language Models" (Faysse et al., 2024)
- "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" (Khattab et al., 2020)
- "ColQwen2: Vision-Language Model for Document Retrieval" (2024)
- "SmolVLM: Small yet Powerful Vision-Language Model" (HuggingFace, 2024)

---

> 📚 **Adaptación al español de la lección [ColPali Vision Native RAG]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).