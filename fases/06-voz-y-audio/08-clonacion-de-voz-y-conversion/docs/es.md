# Clonación de voz y conversión

> Voice cloning (TTS con voz custom): XTTS (6s, 16 idiomas), Tortoise (diffusion, alta calidad), ElevenLabs (closed SOTA). Voice conversion: FreeVC, RVC, So-VITS-SVC. Watermarking y anti-spoofing criticos para etica.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07-texto-a-habla-tts
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar speaker encoder mock.
- Implementar voice conversion mock.
- Evaluar speaker similarity.
- Diagnosticar voice cloning vs voice conversion.

## Constrúyelo

```python
def voice_conversion_mock(source_audio, target_speaker_emb, output_dim=None):
    n_samples = len(source_audio) if output_dim is None else output_dim
    factor = float(target_speaker_emb.mean())
    audio = source_audio * (1 + 0.1 * factor)
    return audio[:n_samples]
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-voice-cloning
fase: 06
leccion: 08
---

1. Default: XTTS (6s, 16 idiomas) o ElevenLabs.
2. Max calidad: Tortoise.
3. Cloning rapido: OpenAI TTS.
4. VC: FreeVC, RVC, So-VITS-SVC.
5. Consentimiento + watermarking + anti-spoofing.
6. Eval: MOS, similarity, WER-ASR.
```

## Ejercicios

1. **XTTS cloning**: clonar tu voz con XTTS en 6s.
2. **RVC conversion**: entrenar RVC con tu voz.
3. **Desafio**: pipeline production de voice cloning
   con XTTS, watermarking con AudioSeal, anti-spoofing
   con RawNet.

## Lecturas recomendadas

- "XTTS" (Casanova et al., 2024)
- "Tortoise" (Betker, 2023)
- Coqui TTS: <https://github.com/coqui-ai/TTS>
- AudioSeal: <https://github.com/facebookresearch/audioseal>

---

> 📚 **Adaptación al español** de la lección "[Voice Cloning & Conversion]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).