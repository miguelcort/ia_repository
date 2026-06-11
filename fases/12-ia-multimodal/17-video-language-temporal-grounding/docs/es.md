# Video language temporal grounding

> Video-language temporal grounding: localizar momentos en video que matchean text query + dense frame sampling (4-32 fps, cap 16-32) + moment retrieval con IoU metric (intersection / union, R1@iou threshold 0.3, 0.5, 0.7) + action segmentation per-frame classification. Models: VideoBERT (Google 2019 seminal), Video-LLaMA (Zhang 2023 +chat Vicuna), VideoChat (Li 2023 +chat StableLM), VideoChat2 (Li 2024 +SOTA +multi-frame), InternVideo (Shanghai AI Lab 2023 +SOTA +video specialized), LLaVA-OV (ByteDance 2024 unified +multi-image +video +SOTA), Qwen2.5-VL (Alibaba 2024 +multilingual +dynamic FPS +SOTA), Video-ChatGPT, Video-LLaVA, Long Video LLM. Frameworks: decord, pyav, transformers, vLLM, SGLang, open_clip. Production: LLaVA-OV + Qwen2.5-VL + InternVideo. Trade-offs: VideoBERT seminal research, Video-LLaMA +chat +simple, VideoChat2 +SOTA +multi-frame, InternVideo +SOTA +video, LLaVA-OV unified, Qwen2.5-VL +multilingual +dynamic. 2025: +Long video + native + reasoning.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/05, 12/06, 12/08
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar sample_dense_frames.
- Implementar temporal_grounding.
- Implementar moment_retrieval_score con IoU.
- Implementar video_text_similarity.
- Diagnosticar modelos SOTA video-language.

## Constrúyelo

```python
def moment_retrieval_score(pred_moments, gt_moments, iou_threshold=0.5):
    def iou(a, b):
        s = max(a[0], b[0])
        e = min(a[1], b[1])
        inter = max(0, e - s)
        union = (a[1] - a[0]) + (b[1] - b[0]) - inter
        return inter / union if union > 0 else 0
    for p in pred_moments:
        for g in gt_moments:
            if iou(p, g) >= iou_threshold:
                return 1
    return 0
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: video-language
fase: 12
leccion: 17
---

1. Temporal grounding.
2. Dense frame sampling.
3. Moment retrieval + IoU.
4. Action segmentation.
5. +SOTA 2024-25.
```

## Ejercicios

1. **InternVideo**: usar
   InternVideo con HuggingFace.
2. **LLaVA-OV**: probar
   LLaVA-OV con video.
3. **Desafio**: video
   retrieval custom.

## Lecturas recomendadas

- "VideoBERT: A Joint Model for Video and Language Representation Learning" (Sun et al., 2019)
- "Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding" (Zhang et al., 2023)
- "VideoChat: Chat-Centric Video Understanding" (Li et al., 2023)
- "InternVideo: General Video Foundation Models via Generative and Discriminative Learning" (Wang et al., 2023)

---

> 📚 **Adaptación al español de la lección [Video Language Temporal Grounding]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).