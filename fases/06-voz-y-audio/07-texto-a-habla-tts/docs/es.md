# Texto a habla (TTS)

> Text -> speech. End-to-end: VITS, VITS2 (SOTA open), XTTS (cloning 6s). Pipeline clasico: Tacotron 2 / FastSpeech 2 + HiFi-GAN vocoder. Aplicaciones: voice assistants, audiobooks, doblaje, accesibilidad.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 05-arquitectura-whisper-y-fine-tuning
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar text -> phonemes mock.
- Generar mel-spec target.
- Vocoder mock (mel-spec -> waveform).
- Pitch extraction basica.

## Constrúyelo

```python
def vocoder_mock(mel_spec):
    n_samples = mel_spec.shape[1] * 256
    return np.random.default_rng(0).normal(size=(n_samples,)).astype(np.float32)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-tts
fase: 06
leccion: 07
---

1. Default: OpenAI TTS, ElevenLabs, Coqui XTTS.
2. Cloning: XTTS (6s audio), Tortoise, ElevenLabs.
3. Multilingual: XTTS, MMS-TTS, Coqui multilingual.
4. Self-host: VITS2, FastSpeech 2, Tacotron 2.
5. Eval: MOS, WER-ASR; latency <200ms streaming.
```

## Ejercicios

1. **Coqui XTTS**: usar XTTS para TTS en espanol.
2. **Voice cloning**: clonar tu voz con XTTS en 6s.
3. **Desafio**: pipeline TTS production con Coqui
   VITS/XTTS, text normalization, eval MOS y WER-ASR.

## Lecturas recomendadas

- "Tacotron 2" (Shen et al., 2018)
- "VITS" (Kim et al., 2021)
- "XTTS" (Casanova et al., 2024)
- Coqui TTS: <https://github.com/coqui-ai/TTS>

---

> 📚 **Adaptación al español** de la lección "[Text-to-Speech]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).