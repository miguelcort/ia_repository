# Long video million token

> Long video understanding: million token context (1M+) con hierarchical compression (multi-level downsample, 3+ niveles, factor 2) + ring attention (split sequence en chunks, N devices, K/V rota, -memory + context) + sliding window (local context 256-512) + keyframe extraction (uniform o scene detection, 16-32 keyframes). Models: LongLLaMA (research +long context), LongVU (+SOTA +video specialized), MovieChat (movies +SOTA), Long Video LLM (+SOTA +video), InternVideo. +Use cases: 1+ hour videos, movies, surveillance, long document. Frameworks: decord, pyav, transformers, vLLM, SGLang, ring-attention-pytorch. Production: LongVU + MovieChat + InternVideo. +Insights: -Long context - compute - memory, +Coverage. Trade-offs: long video - compute + memory, keyframes + compute - coverage, sliding window + simple - coverage, ring + distributed - bandwidth, hierarchical + memory - detail. 2025: +Long + native + reasoning.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/08, 12/17
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar keyframe_extract.
- Implementar hierarchical_compress multi-level.
- Implementar sliding_window_attention.
- Implementar ring_attention_simulation.
- Implementar encode_long_video.
- Diagnosticar long video SOTA.

## Constrúyelo

```python
def hierarchical_compress(video, n_levels=3, factor=2):
    levels = [video]
    cur = video
    for _ in range(n_levels - 1):
        cur = cur[::factor, ::factor, ::factor, :]
        levels.append(cur)
    return levels
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: long-video
fase: 12
leccion: 18
---

1. Million token context.
2. Hierarchical compression.
3. Ring attention.
4. Sliding window.
5. +SOTA 2024-25.
```

## Ejercicios

1. **LongVU**: usar LongVU
   con HuggingFace.
2. **MovieChat**: probar
   MovieChat con movies.
3. **Desafio**: long video
   retrieval custom.

## Lecturas recomendadas

- "LongLLaMA: Scaling LLMs to 1M Context Windows" (Tworkowski et al., 2024)
- "LongVU: Spurious Long-Context Understanding in Video" (Shen et al., 2024)
- "MovieChat: From Dense Token to Sparse Memory for Long Video Understanding" (Song et al., 2024)
- "Ring Attention with Blockwise Transformers for Near-Infinite Context" (Liu et al., 2023)

---

> 📚 **Adaptación al español de la lección [Long Video Million Token]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).