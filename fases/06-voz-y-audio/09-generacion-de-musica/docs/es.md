# Generación de música

> Text-to-music: MusicGen (Meta, open), AudioLDM, Suno/Udio (closed SOTA), Riffusion. Stem separation: Demucs, Spleeter. Aplicaciones: bandas sonoras, content creation, audio ads, demos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07-texto-a-habla-tts
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar piano roll -> audio.
- Generar mel-spectrogram de audio.
- Mock de text-to-music (MusicGen).
- Diagnosticar SOTA music generation.

## Constrúyelo

```python
def piano_roll_mock(notas, duracion=4.0, sample_rate=44100):
    n_frames = int(duracion * sample_rate)
    audio = np.zeros(n_frames)
    for pitch, start, dur in notas:
        start_sample = int(start * sample_rate)
        n_samples = int(dur * sample_rate)
        t = np.arange(n_samples) / sample_rate
        freq = 440.0 * 2 ** ((pitch - 69) / 12)
        audio[start_sample:start_sample + n_samples] = np.sin(2 * np.pi * freq * t)
    return audio
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-music-gen
fase: 06
leccion: 09
---

1. Max calidad: Suno, Udio.
2. Self-host: MusicGen, AudioLDM, Riffusion.
3. Stem sep: Demucs, Spleeter.
4. Real-time: Musika!, Riffusion.
5. Eval: CLAP, FAD, MOS; data licenciada.
```

## Ejercicios

1. **MusicGen demo**: usar MusicGen para generar un
   sample de 10s.
2. **Stem separation**: separar vocals, drums, bass,
   other de una cancion.
3. **Desafio**: pipeline production con MusicGen +
   Demucs + mastering, eval CLAP score en golden set.

## Lecturas recomendadas

- "MusicGen" (Copet et al., 2023)
- "AudioLDM" (Liu et al., 2023)
- AudioCraft: <https://github.com/facebookresearch/audiocraft>

---

> 📚 **Adaptación al español** de la lección "[Music Generation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).