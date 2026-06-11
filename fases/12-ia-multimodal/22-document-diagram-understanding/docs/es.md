# Document diagram understanding

> Document understanding: OCR + layout analysis + VQA. Models: LayoutLMv3 (Microsoft 2022 text + layout + image +modular +SOTA), Donut (Naver 2022 OCR-free +simple seminal), DocVLM (2023), Nougat (Meta 2023 paper->markdown OCR-free +scientific +Math +Tables +Figures +SOTA), GOT (2024 OCR 2.0 +SOTA +efficient), SmolDocling (2024 +efficient), DocOwl2 (2024 +SOTA +open +unified). Use cases: PDFs, forms, tables, charts, scientific papers, receipts, invoices. Frameworks: transformers, vLLM, marker-pdf, docTR, pytesseract. +Production: GOT + DocOwl2 + SmolDocling SOTA. +Insights: +OCR +Layout +VQA. OCR vs OCR-free: OCR-based (Tesseract + LayoutLMv3 + DocVLM +modular +SOTA) vs OCR-free (Donut + Nougat + GOT + SmolDocling +end-to-end +simple). Decision: modular -> LayoutLMv3, simple -> Donut, scientific -> Nougat, SOTA -> GOT o DocOwl2, production -> GOT + DocOwl2 + SmolDocling. 2025: +Native + reasoning + scientific.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/05, 12/06
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar ocr_extract_lines.
- Implementar layout_analysis (header, text, table, figure).
- Implementar table_extract y chart_extract.
- Implementar document_vqa.
- Implementar layoutlmv3_encode y nougat_scientific_paper.
- Diagnosticar OCR vs OCR-free.

## Constrúyelo

```python
def layout_analysis(ocr_lines, image_shape):
    H, W = image_shape[:2]
    regions = []
    for i, line in enumerate(ocr_lines):
        x0, y0, x1, y1 = line["bbox"]
        if y0 < H * 0.1:
            rtype = "header"
        else:
            rtype = "text"
        regions.append({"type": rtype, "bbox": (x0, y0, x1, y1), "text": line["text"]})
    return regions
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: document-vlm
fase: 12
leccion: 22
---

1. OCR + layout + VQA.
2. PDFs, forms, scientific.
3. OCR-free: Donut, Nougat.
4. SOTA: GOT + DocOwl2.
5. +Production 2024-25.
```

## Ejercicios

1. **GOT**: usar GOT con
   HuggingFace.
2. **Nougat**: probar Nougat
   con paper.
3. **Desafio**: DocVLM
   para receipts.

## Lecturas recomendadas

- "LayoutLMv3: Unified Text and Masked Labeling for Document AI" (Huang et al., 2022)
- "OCR-free Document Understanding Transformer" (Kim et al., 2022)
- "Nougat: Neural Optical Understanding for Academic Documents" (Blecher et al., 2023)
- "GOT-OCR2: A Stronger Large Model for OCR" (Wei et al., 2024)

---

> 📚 **Adaptación al español de la lección [Document Diagram Understanding]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).