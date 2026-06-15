# 01 — Fundamentos de audio

> El audio es una señal 1D con estructura temporal, frecuencia y semántica. Entender waveform, espectrograma, y representación digital es prerequisito para Whisper, TTS, y voice agents.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 20-transformada-fourier
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Cargar y visualizar audio (waveform y espectrograma).
- Aplicar STFT, mel-spectrograma, MFCC.
- Diagnosticar sampling rate, bit depth, y canales.
- Usar torchaudio, librosa, o soundfile.

## El problema

El audio es una señal analógica muestreada a digital.
Necesitamos saber cómo se representa, cómo se procesa, y
cómo se transforma para alimentar modelos. La lección
cubre los conceptos: waveform, espectrograma, escala Mel,
MFCC, y los formatos estándar.

## El concepto

**Waveform.** Señal 1D en el tiempo. Sampling rate típico
16kHz (telefonía), 22.05kHz (CDs), 44.1kHz (audio
profesional), 48kHz (video). Bit depth 16 o 24. Canales:
mono (voz) o stereo (música).

**STFT (Short-Time Fourier Transform).** Divide la señal
en ventanas de ~25ms con hop de ~10ms. Aplica FFT a cada
ventana. Resultado: espectrograma 2D (tiempo x frecuencia).
Es la base del análisis de audio moderno.

**Escala Mel.** Escala perceptual: las diferencias de
frecuencia que el oído humano percibe como iguales
corresponden a diferencias iguales en Mel. Se calcula
como `mel = 2595 * log10(1 + freq/700)`. Aplicar banco
de filtros triangulares al espectrograma da el mel-
spectrograma.

**MFCC (Mel-Frequency Cepstral Coefficients).** Aplicar
DCT al mel-spectrograma (log) y quedarse con los primeros
12-13 coeficientes. Es la representación clásica para
reconocimiento de voz. Ha sido reemplazada por mel-
spectrograma crudo en modelos modernos (Whisper).

**Formatos de audio.**

- **WAV:** sin pérdida, PCM. Estándar.
- **FLAC:** sin pérdida, compresión. ~50-70% del WAV.
- **MP3:** con pérdida. Universal.
- **Opus:** moderno, baja latencia, mejor que MP3.
- **M4A/AAC:** Apple, YouTube.

**Pipeline para Whisper.**

1. Cargar audio a 16kHz mono.
2. Calcular log-mel spectrograma (80 canales, ventana
   25ms, hop 10ms).
3. Pad/truncar a 30s (480 frames de mel).
4. Pasar al encoder del modelo.

**Trampas.**

- **Sampling rate incorrecto:** si el modelo espera 16kHz
  y le pasas 44.1kHz, falla. Resamplear siempre.
- **Stereo vs mono:** la mayoría de modelos de voz
  esperan mono. Promediar canales.
- **Silencio al inicio:** el modelo puede confundir
  silencio con pausa. Hacer VAD antes.

## Constrúyelo

```python
import numpy as np


def generate_sine(freq, duration, sr=16000):
    """Genera un seno a una frecuencia dada."""
    t = np.linspace(0, duration, int(sr * duration))
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def stft(signal, n_fft=400, hop=160, win_length=400):
    """STFT simplificada. signal: (T,). Devuelve (F, T_frames)."""
    # Pad para que quepan las ventanas
    pad = n_fft // 2
    padded = np.pad(signal, pad, mode="reflect")
    n_frames = (len(padded) - n_fft) // hop + 1
    window = np.hanning(win_length)
    spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        frame = padded[i * hop:i * hop + n_fft] * window
        spec[:, i] = np.abs(np.fft.rfft(frame))[:n_fft // 2 + 1]
    return spec


def mel_filterbank(n_filters=80, n_fft=400, sr=16000,
                   f_min=0, f_max=8000):
    """Banco de filtros triangulares en escala Mel."""
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = (n_fft / sr) * hz_points
    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(n_filters):
        left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        for j in range(n_fft // 2 + 1):
            if left <= j < center:
                filters[i, j] = (j - left) / (center - left)
            elif center <= j <= right:
                filters[i, j] = (right - j) / (right - center)
    return filters


def mel_spectrogram(signal, sr=16000, n_fft=400, hop=160,
                    n_mels=80):
    """Log-mel spectrograma. (T,) -> (n_mels, T_frames)."""
    spec = stft(signal, n_fft, hop, n_fft)
    filters = mel_filterbank(n_mels, n_fft, sr)
    mel_spec = filters @ spec
    return np.log(np.maximum(mel_spec, 1e-10))
```

## Úsalo

```bash
pip install numpy matplotlib
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio-basics
fase: 06
leccion: 01
---

Eres un asistente que ayuda con preprocesamiento de audio
para un modelo. Recibirás el audio y el modelo. Tu trabajo:

1. Cargar a 16kHz mono (Whisper) o 24kHz (VITS,
   Bark).
2. Resamplear si es necesario.
3. Calcular log-mel spectrograma (80 canales para
   Whisper).
4. Pad/truncar a la longitud esperada.
5. Para TTS: preprocesar con el codec del modelo.
6. Verificar con oído: que no haya clipping ni
   silencios extraños.
```

## Ejercicios

1. **STFT**: implementa y aplica a un archivo de audio.
2. **Mel spectrograma**: visualiza el de una canción.
3. **Desafío**: implementa Griffin-Lim para invertir
   un mel-spectrograma a waveform.

## Lecturas recomendadas

- *Speech and Language Processing* — Jurafsky & Martin.
- *The Scientist and Engineer's Guide to Digital Signal
  Processing* — Steven W. Smith.
- torchaudio: <https://pytorch.org/audio>.
- librosa: <https://librosa.org>.
- Whisper: <https://github.com/openai/whisper>.

---

> 📚 **Adaptación al español** de la lección "[Audio Fundamentals]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
