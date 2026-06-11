# OCR y comprensión de documentos

> Convertir texto en imagenes a texto editable y estructurado. Tesseract, TrOCR, PaddleOCR, LayoutLM, Donut, cloud APIs. Hoy: VLMs generalistas como GPT-4V hacen todo el pipeline de un solo paso.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18-clip-vocabulario-abierto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Preprocesar imagenes para OCR (binarizacion, denoise).
- Detectar lineas de texto via proyeccion.
- Implementar segmentacion por huecos.
- Diagnosticar OCR moderno vs tradicional.

## Constrúyelo

```python
def segmentos_por_huecos(proyeccion, min_gap=2):
    segmentos = []
    in_segmento = False
    start = 0
    gap_count = 0
    for i, val in enumerate(proyeccion):
        if val > 0:
            if not in_segmento:
                start = i
                in_segmento = True
            gap_count = 0
        else:
            if in_segmento:
                gap_count += 1
                if gap_count >= min_gap:
                    segmentos.append((start, i - gap_count + 1))
                    in_segmento = False
    return segmentos
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ocr-elegir
fase: 04
leccion: 19
---

1. Impreso: Tesseract, EasyOCR, PaddleOCR.
2. Manuscritos: TrOCR, HTR-Flor.
3. Estructurados: LayoutLM v3, Donut, Textract.
4. Tablas: TableNet, TableTransformer.
5. PDFs: Nougat, GPT-4V.
```

## Ejercicios

1. **Deskew**: corregir rotacion detectando angulo de
   lineas horizontales.
2. **Tabla a CSV**: detectar intersecciones de lineas y
   mapear a celdas.
3. **Desafio**: implementar pipeline completo de OCR
   con PaddleOCR o TrOCR y comparar CER.

## Lecturas recomendadas

- "TrOCR" (Li et al., 2021)
- "LayoutLM v3" (Huang et al., 2022)
- PaddleOCR: <https://github.com/PaddlePaddle/PaddleOCR>

---

> 📚 **Adaptación al español** de la lección "[OCR and Document Understanding]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).