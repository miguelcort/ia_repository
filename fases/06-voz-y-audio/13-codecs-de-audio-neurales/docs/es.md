# 13 — Codecs neuronales de audio

> Codecs neuronales (EnCodec, SNAC, Mimi) comprimen audio 24kHz a 75 tokens/segundo con calidad perceptual alta. Son la base del streaming speech-to-speech y modelos de audio en tiempo real.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-fundamentos-de-audio
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Entender la diferencia entre codecs clásicos (MP3, Opus)
  y neuronales (EnCodec, SNAC).
- Implementar un codec simple con autoencoder.
- Aplicar EnCodec para compresión y reconstrucción de
  audio.
- Conocer aplicaciones: streaming, voice conversion,
  speech-to-speech.

## El problema

Los codecs clásicos (MP3, Opus, AAC) comprimen audio con
algoritmos de signal processing. Los codecs neuronales
(EnCodec, SNAC, Mimi) usan redes neuronales entrenadas con
loss adversarial para comprimir con calidad perceptual
mucho mejor a bitrates bajos. Son la base del streaming
speech-to-speech (Moshi) y voice agents en tiempo real.

## El concepto

**Codecs clásicos vs neuronales.**

- **Clásicos** (MP3, Opus): algoritmos de signal
  processing con psychoacoustics hand-crafted. Bitrate
  típico 32-128 kbps.
- **Neuronales** (EnCodec, SNAC, Mimi): autoencoder
  con encoder, quantizer, y decoder entrenados
  end-to-end. Bitrate 1.5-24 kbps con calidad similar o
  mejor.

**Arquitectura de un codec neuronal.**

- **Encoder:** CNN que reduce el sample rate 24kHz a
  75 frames/segundo con 128 canales.
- **Quantizer:** residual vector quantizer (RVQ) que
  mapea cada frame a tokens discretos (de un codebook
  aprendido).
- **Decoder:** CNN que reconstruye el waveform desde
  los tokens.

**Modelos canónicos.**

- **EnCodec (Défossez et al., 2022, Meta):** 24 kHz
  bandwidth, ~75 tokens/segundo. SOTA open-source.
- **SNAC (Siuzdak et al., 2024):** multi-scale,
  múltiples bitrates (0.98-22 kbps).
- **Mimi (Kyutai, 2024):** usado en Moshi, base del
  speech-to-speech en tiempo real.
- **DAC (Kumar et al., 2023):** 44.1 kHz para audio
  musical de alta fidelidad.
- **SemantiCodec (Liu et al., 2024):** semantic +
  acoustic tokens para TTS.

**Métricas.**

- **PESQ (Perceptual Eval of Speech Quality):** 1-5,
  mide calidad perceptual.
- **STOI (Short-Time Objective Intelligibility):**
  intelligibilidad.
- **Mel spectral distortion:** distancia en mel
  spectrograma.
- **MOS:** humanos 1-5.

**Aplicaciones.**

- **Streaming speech-to-speech:** Moshi, SpeechGPT.
  Tokens de audio en vez de texto.
- **Voice conversion:** voice conversion en tiempo real
  con codec neuronal.
- **Audio LLMs:** algunos modelos usan codec tokens en
  vez de mel (Moshi).
- **Baja latencia:** ~12.5ms por chunk en streaming.

**Trampas.**

- **Bitrate demasiado bajo:** artefactos audibles. Para
  voz, 6 kbps es el mínimo razonable.
- **Codec incompatible:** diferentes codecs no son
  interoperables. Elige uno y estandariza.
- **Sin fine-tuning:** un codec genérico puede no ser
  óptimo para tu dominio (música vs voz).

## Constrúyelo

```python
import numpy as np


def simple_audio_codec(audio, sr=24000, codebook_size=128,
                       frame_rate=75):
    """Codec de audio simplificado: autoencoder con
    quantization. En producción: EnCodec."""
    frame_size = sr // frame_rate  # samples por frame
    n_frames = len(audio) // frame_size
    # Encoder: cada frame -> un entero del codebook
    frames = audio[:n_frames * frame_size].reshape(n_frames, frame_size)
    # Placeholder: codifica cada frame por su energía promedio
    energies = frames.mean(axis=1)
    quantiles = np.quantile(energies, np.linspace(0, 1, codebook_size + 1))
    tokens = np.digitize(energies, quantiles) - 1
    tokens = np.clip(tokens, 0, codebook_size - 1)
    # Decoder: cada token -> frame con energía reconstruida
    recon_energies = (quantiles[:-1] + quantiles[1:]) / 2
    recon_frames = np.zeros((n_frames, frame_size))
    for i, t in enumerate(tokens):
        recon_frames[i] = recon_energies[t]
    return tokens, recon_frames.flatten(), codebook_size, frame_rate
```

## Úsalo

```bash
pip install encodec
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

Eres un asistente que ayuda con codecs neuronales de
audio. Recibirás el caso de uso. Tu trabajo:

1. Si quieres SOTA: EnCodec (24kHz, 75 tokens/s).
2. Para música: DAC (44.1kHz).
3. Para speech-to-speech en tiempo real: Mimi.
4. Bitrate: 6-12 kbps para voz, 12-24 kbps para música.
5. Streaming: chunk de 12.5ms, latencia ~50ms.
6. Evaluar con PESQ, STOI, MOS.
7. Para deployment: exportar el codec a ONNX o
   TensorRT.
8. Advertir contra bitrates demasiado bajos: artefactos.
```

## Ejercicios

1. **EnCodec**: usa HuggingFace transformers para
   codificar/decodificar audio.
2. **Streaming**: implementa streaming de audio con
   EnCodec.
3. **Desafío**: fine-tunea EnCodec en tu dominio
   (música, voz con ruido).

## Lecturas recomendadas

- *High Fidelity Neural Audio Compression* — Défossez
  et al., 2022 (EnCodec).
- *SNAC* — Siuzdak et al., 2024.
- *Moshi* — Kyutai, 2024.
- *DAC* — Kumar et al., 2023.
- encodec: <https://github.com/facebookresearch/encodec>.

---

> 📚 **Adaptación al español** de la lección "[Neural Audio Codecs]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
