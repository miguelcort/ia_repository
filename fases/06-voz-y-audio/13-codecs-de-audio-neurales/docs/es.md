# Codecs de audio neurales

> Neural networks que comprimen audio a tokens discretos. EnCodec (Meta, RVQ, 24kHz/12kbps), SoundStream (Google), DAC (Descript, 44.1kHz), Lyra (Google, 3kbps telephony). Compression 100-1000x, base de TTS y voice agents.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 12-pipeline-de-asistente-de-voz
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar EnCodec mock con RVQ.
- Decodificar tokens a waveform.
- Calcular bitrate y compression ratio.
- Diagnosticar EnCodec vs DAC vs Lyra.

## Constrúyelo

```python
def encodec_compress_mock(audio, n_codebooks=4, codebook_size=512, sample_rate=24000, hop=320):
    n_frames = len(audio) // hop
    tokens = np.random.default_rng(0).integers(0, codebook_size, size=(n_codebooks, n_frames))
    return tokens, {"bitrate_kbps": 6.0}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio-codec
fase: 06
leccion: 13
---

1. TTS: EnCodec 24kHz (VALL-E, MusicGen).
2. Voice agents: Moshi codec, SoundStream.
3. Music: DAC 44.1kHz.
4. Telephony: Lyra 3kbps.
5. Self-host: encodec, descript-audio-codec, vocos.
6. Compression 100-1000x; PESQ, ViSQOL eval.
```

## Ejercicios

1. **EnCodec usage**: usar EnCodec para comprimir
   audio.
2. **DAC music**: comprimir audio de musica con DAC.
3. **Desafio**: pipeline production con EnCodec para
   TTS / voice agent. Eval PESQ, latency, bitrate.

## Lecturas recomendadas

- "EnCodec" (Defossez et al., 2022)
- "SoundStream" (Zeghidour et al., 2021)
- "DAC" (Kumar et al., 2023)
- "Lyra" (Skoglund et al., 2021)

---

> 📚 **Adaptación al español** de la lección "[Neural Audio Codecs]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).