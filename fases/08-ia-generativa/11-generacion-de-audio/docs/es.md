# Generación de audio

> SOTA: VALL-E/X (codec LM TTS, voice cloning), MusicGen (Meta, text-to-music con EnCodec + T5), AudioLDM 2 (latent diffusion con LLM composer), Bark (multilingual TTS), Stable Audio, Suno/Udio (end-to-end songs). Codec LMs (VALL-E, MusicGen), latent diffusion (AudioLDM), AR (Jukebox), flow matching. Aplicaciones: podcasts, ads, music, accessibility, game audio, deepfake risks.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-voice-and-audio
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar mel spectrogram mock.
- Comparar VALL-E, MusicGen, AudioLDM.
- Diagnosticar codec LMs vs diffusion.
- Evaluar riesgos y aplicaciones.

## Constrúyelo

```python
def mel_spectrogram_mock(audio, n_mels=80, hop=256):
    n = len(audio)
    n_frames = n // hop
    return np.log(np.abs(rng.standard_normal((n_frames, n_mels))) * 0.1 + 1e-6)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio-generation
fase: 08
leccion: 11
---

1. VALL-E: codec LM, voice cloning.
2. MusicGen: EnCodec + T5 + decoder.
3. AudioLDM 2: latent diffusion + LLM.
4. Bark: multilingual TTS.
5. Suno/Udio: end-to-end songs.
```

## Ejercicios

1. **MusicGen**: aplicar MusicGen a
   custom prompt y analizar output.
2. **Bark**: clonar voz con Bark.
3. **Desafio**: entrenar VALL-E-style
   codec LM en custom dataset.

## Lecturas recomendadas

- "VALL-E: Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (Wang et al., 2023)
- "MusicGen: Simple and Controllable Music Generation" (Copet et al., 2023)
- "AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining" (Liu et al., 2024)
- "Bark: Text-Prompted Generative Audio Model" (Suno AI, 2023)

---

> 📚 **Adaptación al español** de la lección "[Audio Generation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).