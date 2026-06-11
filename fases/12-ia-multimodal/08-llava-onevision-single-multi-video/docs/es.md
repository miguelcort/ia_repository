# LLaVA-OneVision single multi video

> LLaVA-OneVision (Li 2024, ByteDance): single model unified image + multi-image + video con sub-image tiling (aspect ratio preservado, 1-4 tiles per image) + frame sampling temporal (uniform 8-16 frames standard) + OneVision projection (single layer). Variants: LLaVA-OV 0.5B, 7B, 72B. +SOTA image, multi-image, video 2024-25. Frame sampling strategies: uniform (linspace(0, T-1, n), +coverage, +deterministic, standard), random (choice sin replacement, +diversity, training), dense (primeros n), sparse (cada k). Multi-image encoding: multiple images -> each -> 1+ tiles -> concat tokens -> +cross-image reasoning (compare, spatial, temporal). Variants: LLaVA-OV, Molmo (multi-image + pointing), Idefics2 (multi-image + SigLIP), InternVL3. Frameworks: transformers (HF), vLLM, SGLang, open_clip, decord. +Use cases: VQA, multi-doc, multi-frame, image comparison, video chat. Production: LLaVA-OV + InternVL3 + Qwen-VL. Trade-offs: LLaVA-OV + unified + SOTA, LLaVA-Next any res single image, Video-LLaVA + video specialized.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/05, 12/06, 12/07
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar sample_video_frames (uniform, random).
- Implementar onevision_tile_image.
- Implementar onevision_tile_video.
- Implementar onevision_forward unified.
- Diagnosticar trade-offs LLaVA-OV vs LLaVA-Next vs Video-LLaVA.

## Constrúyelo

```python
def sample_video_frames(video, n_frames=8, strategy="uniform"):
    T = video.shape[0]
    if strategy == "uniform":
        idx = np.linspace(0, T - 1, n_frames).astype(int)
    return video[idx]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: llava-ov
fase: 12
leccion: 08
---

1. Single unified model.
2. Sub-image tiling.
3. Frame sampling 8-16.
4. Multi-image + video.
5. +SOTA 2024-25.
```

## Ejercicios

1. **LLaVA-OV**: usar LLaVA-OV
   con HuggingFace.
2. **Video-LLaVA**: probar
   Video-LLaVA con video.
3. **Desafio**: multi-image
   reasoning custom.

## Lecturas recomendadas

- "LLaVA-OneVision: Easy Visual Task Transfer" (Li et al., 2024)
- "Video-LLaVA: Learning United Visual Representation by Alignment Before Projection" (Lin et al., 2023)
- "VideoChat: Chat-Centric Video Understanding" (Li et al., 2023)
- "Long Video Understanding with VideoLLaMA" (Zhang et al., 2023)

---

> 📚 **Adaptación al español de la lección [LLaVA OneVision Single Multi Video]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).