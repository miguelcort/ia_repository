# 12 — Video understanding pipeline

> Video understanding: frame extraction, ASR (Whisper), scene detection, temporal grounding, video QA. Modelos: Video-LLaVA, Gemini 1.5 Pro (1M context), GPT-4o, Qwen2-VL. Use cases: highlight generation, action recognition, video search.

**Tipo:** Capstone
**Lenguajes:** Python
**Prerrequisitos:** Fase 12 (multimodal), Fase 6 (audio), Fase 11
**Tiempo estimado:** 25 horas

## Objetivos

- Frame extraction + scene detection.
- ASR con Whisper.
- Video QA con VLM.
- Eval sobre VideoMME, EgoSchema.

## El problema

Video understanding pipeline: (1) Frame extraction
(PyAV, ffmpeg, 1 fps). (2) Scene detection
(TransNetV2, PySceneDetect). (3) ASR con Whisper
(timestamps). (4) OCR (subtitle text, on-screen
text). (5) Embedding: Video-LLaVA, ImageBind.
(6) VLM: Gemini 1.5 Pro (1M context), GPT-4o,
Qwen2-VL. Use cases: video search, highlight
generation, action recognition. Eval: VideoMME,
EgoSchema, NExT-QA.

## Constrúyelo

```python
import av
from transformers import Qwen2VLForConditionalGeneration


def extract_frames(video_path, fps=1):
    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        if frame.time >= len(frames) / fps:
            frames.append(frame.to_image())
    return frames


def video_qa(video_path, question, model="Qwen/Qwen2-VL-7B"):
    vlm = Qwen2VLForConditionalGeneration.from_pretrained(model)
    return vlm.chat(video=video_path, question=question)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-video-understanding
fase: 19
leccion: 12
---

1. Frame extraction.
2. ASR + scene detection.
3. VLM (Qwen2-VL, Gemini 1.5).
4. Temporal grounding.
5. Eval VideoMME.
```

## Ejercicios

1. **Frame + ASR**: 10 videos
   de YouTube.
2. **VLM QA**: 100 preguntas
   sobre videos.
3. **Desafío**: highlight
   generation pipeline.

## Lecturas recomendadas

- "Qwen2-VL" (Alibaba 2024)
- "Gemini 1.5 Pro" (Google 2024)
- "VideoMME" (Fu 2024)
- "Video-LLaVA" (Lin 2023)

---

> 📚 **Adaptación al español** de la lección
> "[12-video-understanding-pipeline]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
