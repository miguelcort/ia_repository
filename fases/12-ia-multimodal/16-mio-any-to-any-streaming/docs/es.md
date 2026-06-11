# MIO any-to-any streaming

> MIO (Wang 2024): foundation model on multimodal tokens con any-to-any capabilities (text->text, image->text, text->image, audio->text, video->text, etc.) + streaming generation (chunks de 64 tokens, decode incremental, -TTFT latency, +UX) + encoder-decoder architecture. +SOTA 2024-25. Variants: 7B, 13B. Vocab: 133248 (text + image VQ + audio + video). Frameworks: original, transformers (HF), vLLM, SGLang. +Insights: -Single modality, +Any-to-any, +Streaming. Production: MIO +SOTA. Streaming: split tokens en chunks, decode chunk por chunk, emit output incremental, -latency, +UX, real-time, video chat, audio, live translation. Family: MIO (any-to-any streaming +SOTA), Emu3 (AR +simple VQ-VAE 32768), TransFusion (AR+diffusion +quality), Show-o (discrete diffusion +simple), Chameleon (early fusion VQ-VAE 8192 seminal), Janus-Pro (decoupled +specialized). Decision: any-to-any -> MIO, simple -> Emu3, quality -> TransFusion, diffusion -> Show-o, specialized -> Janus-Pro. Production: MIO + TransFusion + Show-o + Janus-Pro SOTA. 2025: +Native + any-to-any + reasoning + video.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/11, 12/12, 12/13, 12/14
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar mio_tokenize para text/image/audio/video.
- Implementar mio_streaming_chunk.
- Implementar mio_streaming_decode.
- Implementar mio_any_to_any.
- Diagnosticar streaming vs batch.

## Constrúyelo

```python
def mio_tokenize(modality, data, vocab_size=133248):
    if modality == "image":
        rng = np.random.default_rng(hash(data.tobytes()[:64]) & 0xFFFFFFFF)
        H, W, C = data.shape
        n = (H // 16) * (W // 16)
        return rng.integers(0, 32768, size=n) + 100480
    elif modality == "video":
        return mio_tokenize("image", data[0])  # simplified
    # ...
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: mio
fase: 12
leccion: 16
---

1. Any-to-any.
2. Streaming chunks.
3. 4 modalities.
4. +SOTA 2024-25.
5. +UX +TTFT.
```

## Ejercicios

1. **MIO**: usar MIO
   con HuggingFace.
2. **Streaming**: implementar
   streaming custom.
3. **Desafio**: MIO para
   multimodal agent.

## Lecturas recomendadas

- "MIO: A Foundation Model on Multimodal Tokens" (Wang et al., 2024)
- "StreamingLLM: Attention Sinks for Infinite Length Inference" (Xiao et al., 2023)
- "vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023)

---

> 📚 **Adaptación al español de la lección [MIO Any-to-Any Streaming]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).