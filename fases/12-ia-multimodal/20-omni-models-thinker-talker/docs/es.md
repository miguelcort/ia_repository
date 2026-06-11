# Omni models thinker talker

> Omni models: thinker + talker architecture. Thinker: multi-modal encoder process text + audio + image + video -> unified hidden states (transformer backbone). Talker: hidden -> text logits (vocab 200K) + audio codes (4096 codebook). Real-time streaming <200ms latency. +SOTA 2024-25. Models: GPT-4o (OpenAI 2024 seminal real-time), Qwen2.5-Omni (Alibaba 2024 +open +multilingual +SOTA 7B-72B), Moshi (Kyutai 2024 +SOTA speech 200ms +open), Gemini 1.5 Pro (Google 2024 +SOTA +long context +multimodal), Reka (2024 +multimodal +chat +SOTA). Frameworks: openai, transformers (HF), vLLM, SGLang, dashscope. +Insights: -Single modality, +Unified, +Streaming, +Real-time. Production: GPT-4o + Qwen2.5-Omni + Moshi SOTA. +Use cases: voice chat, video chat, real-time, multimodal, multilingual. Trade-offs: Omni + unified + streaming, separate models specialized. 2025: +Real-time + native + reasoning + speech.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/16, 12/19
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar omni_thinker multi-modal.
- Implementar omni_talker text + audio.
- Implementar omni_streaming_first_token.
- Implementar gpt4o_style_forward.
- Diagnosticar GPT-4o vs Qwen2.5-Omni vs Moshi.

## Constrúyelo

```python
def omni_thinker(modalities, embed_dim=4096):
    parts = []
    for mod, data in modalities:
        if mod == "text":
            t = np.random.default_rng(hash(data) & 0xFFFFFFFF).standard_normal((len(data.split()), embed_dim)) * 0.1
            parts.append(t)
    return np.concatenate(parts, axis=0)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: omni-models
fase: 12
leccion: 20
---

1. Thinker + talker.
2. 5 modalities.
3. Real-time <200ms.
4. 200K text + 4096 audio.
5. +SOTA 2024-25.
```

## Ejercicios

1. **Qwen2.5-Omni**: usar
   Qwen2.5-Omni con dashscope.
2. **Moshi**: probar Moshi
   speech-to-speech.
3. **Desafio**: omni custom
   para voice agent.

## Lecturas recomendadas

- "GPT-4o System Card" (OpenAI, 2024)
- "Qwen2.5-Omni Technical Report" (Xu et al., 2024)
- "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue" (Kyutai Labs, 2024)
- "Gemini 1.5: Unlocking Multimodal Understanding Across Millions of Tokens of Context" (Reid et al., 2024)

---

> 📚 **Adaptación al español de la lección [Omni Models Thinker Talker]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).