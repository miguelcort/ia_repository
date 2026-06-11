# Qwen-VL family dynamic FPS

> Qwen-VL family (Alibaba 2023-2024): Qwen-VL (Bai 2023, seminal +multilingual +OCR 7B), Qwen2-VL (Wang 2024, dynamic resolution + dynamic FPS video + 2D-RoPE position encoding), Qwen2.5-VL (Bai 2024, +advanced 2B-72B +SOTA). 2D-RoPE: position encoding 2D (half dim para h, half para w, standard RoPE por axis, sin/cos pairs base 10000) + +spatial + +generalization any res. Dynamic FPS sampling: video -> frames a target FPS + max frames cap (8-16) + +coverage + -compute. +Variants: Qwen2-VL dynamic, LLaVA-OV 8 frames, Video-ChatGPT 100 frames, Long Video 16+ frames. Frameworks: transformers (HF), vLLM, SGLang, open_clip, decord, pyav. +Use cases: VLM, OCR, video, multilingual. +Production: Qwen2.5-VL SOTA. +Comparado: Qwen-VL seminal, Qwen2-VL dynamic, Qwen2.5-VL advanced. SOTA 2024-25 es mix Qwen2.5-VL + InternVL3 + Molmo.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/05, 12/06, 12/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar dynamic_resolution_tiles.
- Implementar dynamic_fps_sampling.
- Encode image tiles y video frames.
- Implementar 2D-RoPE position.
- Diagnosticar familia Qwen-VL.

## Constrúyelo

```python
def dynamic_resolution_tiles(image, max_tiles=4, min_tiles=1):
    H, W, C = image.shape
    aspect = W / H
    best = (1, 1)
    for n in range(min_tiles, max_tiles + 1):
        for nh in range(1, n + 1):
            nw = n // nh
            if nh * nw != n:
                continue
            score = abs(np.log((nw / nh) / aspect))
            if score < best_score:
                best = (nh, nw)
    return crop_tiles(image, best)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: qwen-vl
fase: 12
leccion: 09
---

1. Qwen-VL, Qwen2-VL, Qwen2.5-VL.
2. Dynamic res + dynamic FPS.
3. 2D-RoPE position.
4. +Multilingual + OCR.
5. +SOTA 2024-25.
```

## Ejercicios

1. **Qwen2-VL**: usar
   Qwen2-VL con HuggingFace.
2. **Qwen2.5-VL**: probar
   Qwen2.5-VL dynamic FPS.
3. **Desafio**: Qwen2.5-VL
   para custom VLM.

## Lecturas recomendadas

- "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond" (Bai et al., 2023)
- "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution" (Wang et al., 2024)
- "Qwen2.5-VL Technical Report" (Bai et al., 2024)

---

> 📚 **Adaptación al español de la lección [Qwen-VL Family Dynamic FPS]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).