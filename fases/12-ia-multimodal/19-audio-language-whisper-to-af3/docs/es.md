# Audio language Whisper to AF3

> Audio-language models: Whisper (Radford 2023, OpenAI, ASR encoder-decoder transformer, mel spectrogram 80 mels, 30s chunks @ 16kHz, 99 langs, 680K hours, 5 sizes tiny 39M - large 1550M, +multilingual +SOTA seminal), Bark/Suno (TTS +music +SFX), AudioPaLM (Google 2023 speech-text unified +translation), Qwen-Audio (Alibaba 2023 +multimodal +multilingual +SOTA), AudioLM (Google 2023 audio continuation semantic + acoustic tokens), MusicGen (Meta 2023 music +text +melody), Audio Flamingo 3 (AF3 2024 unified ASR + TTS + understanding + generation + audio + text + image + few-shot in-context +SOTA 2024-25, 1B-7B). Frameworks: original, transformers (HF), vLLM, SGLang. +Insights: -Single task, +Unified, +SOTA. +Production: AF3 + Qwen-Audio + Qwen2-Audio SOTA. +Use cases: ASR, TTS, audio chat, music, multimodal, few-shot. Decision: ASR -> Whisper, TTS -> Bark, multimodal -> Qwen-Audio, music -> MusicGen, unified -> AF3. 2025: +Unified + native + reasoning + speech.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12/05, 12/06
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar mel_spectrogram_features.
- Implementar whisper_encoder y whisper_decoder.
- Implementar audio_text_similarity.
- Implementar audio_lm_forward.
- Implementar af3_unified.
- Diagnosticar Whisper vs Qwen-Audio vs AF3.

## Constrúyelo

```python
def whisper_encoder(audio, n_mels=80, d_model=512):
    feats = mel_spectrogram_features(audio, n_mels=n_mels)
    rng = np.random.default_rng(hash(audio.tobytes()[:64]) & 0xFFFFFFFF)
    return rng.standard_normal((feats.shape[0], d_model)) * 0.1
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: audio-language
fase: 12
leccion: 19
---

1. Whisper ASR.
2. Bark TTS.
3. Qwen-Audio multimodal.
4. AF3 unified.
5. +SOTA 2024-25.
```

## Ejercicios

1. **Whisper**: usar Whisper
   con transformers.
2. **Qwen-Audio**: probar
   Qwen-Audio multimodal.
3. **Desafio**: AF3 para
   audio chat.

## Lecturas recomendadas

- "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2023)
- "Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models" (Chu et al., 2023)
- "AudioPaLM: A Large Language Model That Can Speak and Listen" (Rubenstein et al., 2023)
- "Audio Flamingo 3: Advancing Audio Understanding with Multi-modal Large Language Model" (2024)

---

> 📚 **Adaptación al español de la lección [Audio Language Whisper to AF3]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).