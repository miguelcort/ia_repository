# Modelos de lenguaje de audio

> LLMs que entienden y generan audio. Qwen2-Audio (audio understanding), VALL-E (zero-shot TTS via audio tokens), Moshi (full-duplex speech-to-speech). Audio encoder + LLM backbone + audio output.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09-generacion-de-musica
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar codificacion de audio a tokens (EnCodec mock).
- Generar embeddings multimodales (CLAP mock).
- Calcular CLAP similarity.
- Diagnosticar audio understanding vs TTS.

## Constrúyelo

```python
def encode_audio_tokens(audio, n_codebooks=4, codebook_size=512):
    n_frames = len(audio) // 320
    rng = np.random.default_rng(0)
    return rng.integers(0, codebook_size, size=(n_codebooks, n_frames))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio-llm
fase: 06
leccion: 10
---

1. Audio understanding: Qwen2-Audio, SALMONN.
2. Zero-shot TTS: VALL-E 2, XTTS.
3. Speech-to-speech: Moshi, GPT-4o voice.
4. Custom: Whisper + Qwen2-7B fine-tune.
5. Frameworks: ms-swift, OpenOmni, transformers.
6. SFT + DPO, audio captioning + QA.
```

## Ejercicios

1. **Qwen2-Audio**: usar Qwen2-Audio para audio
   captioning.
2. **VALL-E demo**: usar VALL-E para zero-shot TTS.
3. **Desafio**: pipeline production con Whisper encoder +
   Qwen2-7B LoRA, SFT en AIR-Bench, deploy en
   production.

## Lecturas recomendadas

- "Qwen2-Audio" (Chu et al., 2024)
- "VALL-E" (Wang et al., 2023)
- "SALMONN" (Tang et al., 2024)
- "Moshi" (Kyutai, 2024)

---

> 📚 **Adaptación al español** de la lección "[Audio Language Models]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).