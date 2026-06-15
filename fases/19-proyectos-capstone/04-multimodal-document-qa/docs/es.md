# 04 — Multimodal document QA

> Multimodal document QA: PDFs, slides, imágenes con texto, tablas, charts. LayoutLMv3, DocFormer, GPT-4V, Claude 3.5 Sonnet. Pipeline: OCR + layout detection + chunking + retrieval + LLM synthesis. Citations verificables.

**Tipo:** Capstone
**Lenguajes:** Python (ingestión), TypeScript (UI)
**Prerrequisitos:** Fase 12 (multimodal), Fase 11 (LLM)
**Tiempo estimado:** 25 horas

## Objetivos

- Pipeline multimodal: PDF → OCR + layout + chunks.
- Retrieval con citations verificables.
- Manejo de tablas, charts, figuras.
- Evaluar faithfulness y citation accuracy.

## El problema

Documentos reales son multimodales: texto, tablas,
figuras, charts, ecuaciones. RAG naive sobre
texto extraído pierde estructura. Solución: (1) PDF
parsing con layout detection (DocLayNet, DiT).
(2) Table extraction (TableTransformer, Nougat).
(3) Chart understanding (GPT-4V). (4) Multi-modal
embeddings (ColPali, ColQwen). Citations son
esenciales: cada respuesta debe referenciar página
y bbox.

## Constrúyelo

```python
from colpali_engine.models import ColPali


def colpali_retrieval(pdf_pages, query):
    """ColPali: multi-vector retrieval sobre PDF pages.
    No necesita OCR."""
    model = ColPali.from_pretrained("vidore/colpali-v1.2")
    page_embeddings = model.encode(pdf_pages)
    query_emb = model.encode([query])
    scores = page_embeddings @ query_emb.T
    return pdf_pages[scores.argmax()]


def multimodal_synthesis(query, retrieved_pages, llm):
    """Síntesis multimodal con GPT-4V."""
    images = [page.to_image() for page in retrieved_pages]
    return llm.chat(query, images=images)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-multimodal-qa
fase: 19
leccion: 04
---

1. PDF parsing con layout.
2. ColPali o LayoutLM.
3. GPT-4V synthesis.
4. Citations page + bbox.
5. Eval faithfulness.
```

## Ejercicios

1. **ColPali**: ingestar 100 PDFs,
   benchmark retrieval.
2. **GPT-4V QA**: implementar pipeline
   con citations.
3. **Desafío**: tabla extraction
   + chart understanding.

## Lecturas recomendadas

- "ColPali" (Faysse 2024)
- "GPT-4V System Card" (OpenAI 2023)
- "DocLayNet" (IBM 2022)
- "Nougat" (Meta 2023)

---

> 📚 **Adaptación al español** de la lección
> "[04-multimodal-document-qa]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
