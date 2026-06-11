# Any resolution patch-n-pack

> Any resolution image encoding: split image en sub-tiles (1, 2, 4, 9, 16) preservando aspect ratio (grid selection minimiza |log(n_w/n_h) - log(W/H)|), resize cada tile a target size (336x336 o 448x448), ViT encode cada tile por separado, concat todos los tokens en una secuencia. +Thumbnail (1 tile global low res) + tiles (detail). +Any resolution, +OCR, +document, +high detail, +SOTA. Variants: LLaVA-Next (Liu 2024, 1-4 tiles, 336px, thumbnail + tiles), Qwen-VL (Alibaba 2024, dynamic FPS + dynamic resolution, +multilingual), InternVL3 (Shanghai AI Lab 2024, dynamic, +multilingual, 4B-72B), Molmo (AI2 2024, +advanced +OCR). Frameworks: transformers (HF), vLLM, open_clip, timm. Hoy: SOTA production standard 2024-25. +Use cases: OCR, document understanding, VQA, multilingual. Trade-offs: any-res + OCR + high res + compute, fixed + simple + fast - OCR. Production: LLaVA-Next + InternVL3 + Qwen-VL.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/01, 12/05
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar select_best_grid preservando aspect ratio.
- Implementar split_into_tiles y resize_tile.
- Implementar pack_tiles_with_thumbnail.
- Implementar pack_for_llm (concat con ViT).
- Diagnosticar variantes SOTA.

## Constrúyelo

```python
def select_best_grid(image_h, image_w, max_tiles=4, min_tiles=1):
    aspect = image_w / image_h
    best = (1, 1)
    best_score = float("inf")
    for n in range(min_tiles, max_tiles + 1):
        for nh in range(1, n + 1):
            nw = n // nh
            if nh * nw != n:
                continue
            score = abs(np.log((nw / nh) / aspect))
            if score < best_score:
                best = (nh, nw)
                best_score = score
    return best
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: any-resolution
fase: 12
leccion: 06
---

1. Grid selection.
2. Split + thumbnail.
3. Resize + ViT.
4. Concat tokens.
5. +OCR, +SOTA.
```

## Ejercicios

1. **LLaVA-Next**: probar
   LLaVA-Next con any-res.
2. **Qwen-VL**: usar Qwen-VL
   dynamic FPS.
3. **Desafio**: any-res para
   OCR custom.

## Lecturas recomendadas

- "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge" (Liu et al., 2024)
- "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond" (Bai et al., 2023)
- "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks" (Chen et al., 2024)

---

> 📚 **Adaptación al español de la lección [Any Resolution Patch-n-Pack]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).