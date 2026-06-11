# Fundamentos de audio

> Audio digital: muestreo de presion sonora. Sample rate (16kHz voz, 44.1kHz audio), bit depth (16-bit), representacion 1D. Features: RMS, ZCR, pitch, mel-spectrogram, MFCC. Para ML: mel-spec como input a Whisper, wav2vec, AST.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-fundamentos-ml/01-pandas-y-numpy
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Generar onda sinusoidal.
- Validar sample rate.
- Calcular RMS energy y ZCR.
- Diagnosticar cuando usar cada sample rate.

## Constrúyelo

```python
def generar_seno(frecuencia, duracion, sample_rate=16000, amplitud=1.0):
    t = np.arange(0, duracion, 1 / sample_rate)
    return amplitud * np.sin(2 * np.pi * frecuencia * t)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio
fase: 06
leccion: 01
---

1. Carga: librosa.load(path, sr=16000, mono=True).
2. Features: mel-spec, MFCC, spectrogram.
3. Augmentation: time stretch, pitch shift, noise.
4. Modelos: Whisper, wav2vec, AST.
5. 16kHz mono para ASR, 80 mels.
```

## Ejercicios

1. **Resampling**: implementar resamplear audio de
   44100 a 16000 Hz.
2. **Mel filter bank**: implementar conversion Hz a Mel
   y construir el mel filter bank.
3. **Desafio**: pipeline completo de carga y
   preprocesamiento de audio, generar mel-spec
   visualizable.

## Lecturas recomendadas

- librosa: <https://librosa.org/>
- torchaudio: <https://pytorch.org/audio/>
- "Speech and Language Processing" (Jurafsky & Martin)

---

> 📚 **Adaptación al español** de la lección "[Audio Fundamentals]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).