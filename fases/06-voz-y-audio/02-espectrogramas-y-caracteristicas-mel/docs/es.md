# Espectrogramas y características mel

> Audio como imagen 2D: mel-spectrogram (tiempo x mel-freq). Escala Mel (perceptual). 80 mels para Whisper, 128 para TTS. MFCC clasico, mel-spec moderno. Librerias: librosa, torchaudio.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-fundamentos-de-audio
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar FFT y STFT.
- Implementar conversion Hz-Mel.
- Construir mel filter bank.
- Generar mel-spectrogram completo.

## Constrúyelo

```python
def mel_spectrogram(senal, sample_rate=16000, n_mels=80, n_fft=400, hop=160):
    spec = stft_simple(senal, ventana=n_fft, hop=hop, n_fft=n_fft)
    filters = mel_filterbank(n_mels=n_mels, n_fft=n_fft, sample_rate=sample_rate)
    mel_spec = filters @ spec
    return np.log(mel_spec + 1e-9)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio-features
fase: 06
leccion: 02
---

1. ASR: 80 mels, n_fft 400, hop 160, Whisper.
2. Clasico: 13 MFCC + deltas + CMN.
3. Audio clasif: mel-spec 64-128, SpecAugment.
4. Music: 96-128 mels.
5. TTS: 80-128 mels target, vocoder.
6. 16kHz voz, 22-44kHz musica.
```

## Ejercicios

1. **STFT real**: usar librosa.stft en vez de la
   implementacion naive.
2. **MFCC completo**: implementar pipeline STFT -> mel ->
   log -> DCT.
3. **Desafio**: visualizador de mel-spectrogram con
   librosa.display.specshow, comparar senales de voz
   vs musica.

## Lecturas recomendadas

- librosa.feature.melspec: <https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html>
- torchaudio.transforms: <https://pytorch.org/audio/stable/transforms.html>
- "Speech and Audio Signal Processing" (Gold & Morgan)

---

> 📚 **Adaptación al español** de la lección "[Spectrograms and Mel Features]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).