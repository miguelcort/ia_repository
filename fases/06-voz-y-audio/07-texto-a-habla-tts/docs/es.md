# 07 — Texto a voz (TTS)

> TTS (Text-to-Speech) convierte texto en habla sintética. Tacotron, FastSpeech, VITS, y Bark son los modelos canónicos. ElevenLabs y OpenAI son los SOTA comerciales.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-fundamentos-de-audio
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar un pipeline TTS: texto → tokens → mel →
  waveform.
- Aplicar modelos pre-entrenados: Tacotron, FastSpeech,
  VITS, Bark.
- Diagnosticar trade-offs entre naturalidad y latencia.
- Conocer SOTA comercial: ElevenLabs, OpenAI TTS, Google
  Cloud TTS.

## El problema

TTS convierte texto en habla sintética. Las
aplicaciones van desde asistentes de voz (Alexa, Siri)
hasta doblaje, accesibilidad, y voice cloning. Los modelos
modernos son neuronales (end-to-end), reemplazando los
sistemas concatenativos clásicos. La lección cubre la
evolución y el pipeline típico.

## El concepto

**Pipeline TTS clásico (Tacotron 2, 2017).**

1. **Text encoder:** convierte caracteres/tokens en
   embeddings (con convs y BiLSTM).
2. **Attention-based decoder:** autoregresivamente
   genera frames de mel-spectrograma, attendiendo al
   text encoder.
3. **Vocoder (WaveNet / WaveGlow):** convierte el mel-
   spectrograma en waveform.

**FastSpeech (Ren et al., 2020).** Non-autoregressive:
usa duración predicha por un modelo separado. Más rápido
(10-100x) y más estable que Tacotron, pero requiere
alineación fonema-duración.

**VITS (Kim et al., 2021).** End-to-end con VAE +
adversarial training. Genera waveform directamente desde
text, sin mel intermedio. Excelente calidad.

**Bark (Suno, 2023).** Transformer generativo tipo
GPT que produce audio crudo. Soporta multilingual,
emociones, y efectos (risa, música). Open source.

**XTTS (Coqui, 2024).** Voice cloning zero-shot con
solo 6 segundos de audio de referencia. Multilingual.

**ElevenLabs / OpenAI TTS / Google Cloud TTS.** SOTA
comercial, calidad indistinguishable de humanos. Caro.
Latencia ~200-500ms para streaming.

**Métricas.**

- **MOS (Mean Opinion Score):** humanos puntúan 1-5.
  SOTA comercial: 4.5+. Open source: 3.5-4.2.
- **WER con ASR:** transcribir el output TTS con Whisper
  y medir WER vs original. Mide intelligibilidad.
- **SIM (Speaker Similarity):** cosine similarity con
  embedding del hablante de referencia.
- **Latencia:** tiempo desde texto hasta audio.

**Trampas.**

- **Robots, números, símbolos:** "Dr. Smith compró 3
  libros a $9.99" puede sonar mal. Pre-procesar texto
  (text normalization).
- **Pronunciación de nombres propios:** agregar lexicon
  o usar g2p (grapheme-to-phoneme) modelo.
- **Discontinuidades en audio:** artefactos al cambiar
  de fonema. FastSpeech y VITS son más suaves que
  Tacotron.

## Constrúyelo

```python
import numpy as np


def text_to_phonemes(text):
    """Conversión texto a fonemas simplificada.
    En producción: espeak, g2p, o multilingual TTS frontend."""
    # Mapeo trivial
    text_lower = text.lower()
    # Aquí iría un lexicon o un modelo g2p
    return text_lower.split()


def mel_spectrogram(phonemes, n_mels=80, n_frames=200):
    """Genera mel-spectrograma fake. En producción: tacotron/vits."""
    # Placeholder: mel-spectrograma constante
    return np.random.randn(n_mels, n_frames).astype(np.float32) * 0.1


def vocoder(mel_spec):
    """Convierte mel a waveform. En producción: WaveNet, HiFi-GAN."""
    n_samples = mel_spec.shape[1] * 256
    return np.random.randn(n_samples).astype(np.float32) * 0.05


def tts_pipeline(text, sample_rate=24000):
    """Pipeline TTS simplificado."""
    phonemes = text_to_phonemes(text)
    mel = mel_spectrogram(phonemes)
    waveform = vocoder(mel)
    return waveform, sample_rate
```

## Úsalo

```bash
pip install TTS
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

Eres un asistente que ayuda con text-to-speech. Reci-
birás el texto, el caso de uso, y la calidad objetivo. Tu
trabajo:

1. Si quieres SOTA: ElevenLabs, OpenAI TTS, Google
   Cloud TTS.
2. Si open-source: VITS, XTTS, Bark, Coqui TTS.
3. Para voice cloning: XTTS con 6s de referencia.
4. Para real-time (latencia < 200ms): streaming
   TTS (VITS streaming, ElevenLabs).
5. Pre-procesar texto: text normalization, g2p.
6. Evaluar con MOS, WER con ASR, latencia.
7. Advertir contra voice cloning sin consentimiento
   (ético y legal).
```

## Ejercicios

1. **Coqui TTS**: usa Coqui TTS para generar audio en
   español.
2. **Voice cloning**: clona tu voz con XTTS.
3. **Desafío**: implementa streaming TTS con VITS.

## Lecturas recomendadas

- *Tacotron 2* — Shen et al., 2017.
- *FastSpeech* — Ren et al., 2020.
- *VITS* — Kim et al., 2021.
- *Bark* — Suno, 2023.
- Coqui TTS: <https://github.com/coqui-ai/TTS>.
- ElevenLabs: <https://elevenlabs.io>.

---

> 📚 **Adaptación al español** de la lección "[Text-to-Speech (TTS)]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
