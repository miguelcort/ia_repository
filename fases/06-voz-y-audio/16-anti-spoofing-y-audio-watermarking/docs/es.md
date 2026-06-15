# 16 — Anti-spoofing y marcas de agua en audio

> Detectar audio sintético (anti-spoofing) y marcar audio generado con watermarks invisibles (SynthID, AudioSeal) son esenciales para combatir deepfakes y desinformación.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 06-reconocimiento-de-hablante-y-verificacion,
                  08-clonacion-de-voz-y-conversion
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar un detector de audio sintético (RawNet2,
  AASIST).
- Aplicar watermarks en audio (SynthID, AudioSeal).
- Diagnosticar el trade-off entre calidad y capacidad de
  detección.
- Conocer las implicaciones éticas y regulatorias.

## El problema

La generación de audio por IA es tan buena que es
indistinguible de humanos. Eso tiene dos consecuencias: (1)
deepfakes y fraude (voz que simula ser CEO llamando por
teléfono), y (2) necesidad de detección (anti-spoofing) y
marcado (watermarks). La lección cubre ambas técnicas.

## El concepto

**Anti-spoofing.** Detectar si un audio es real o
generado. Dos enfoques:

- **Basado en features:** extraer features que difieren
  entre real y sintético (e.g. high-frequency artifacts,
  phase inconsistencies, breathing patterns).
- **End-to-end:** redes neuronales (RawNet2, AASIST)
  que clasifican waveform crudo o mel-spectrograma.

**Modelos SOTA.**

- **RawNet2 (Jung et al., 2020):** SOTA en ASVspoof.
  CNN sobre waveform crudo.
- **AASIST (Jung et al., 2021):** attention-based. SOTA
  en ASVspoof 2021.
- **WaveFake:** detector de speech synthesis.
- **AudioSeal (Meta, 2024):** SOTA open-source.
  Detecta audio de EnCodec, MusicGen, AudioLDM.
- **SynthID (Google, 2024):** watermarking para
  producción comercial.

**Watermarking.** Marcar audio generado con una señal
invisible que sobrevive a compresión y edición:

- **AudioSeal:** watermark imperceptible (16 kbps
  payload) que sobrevive a MP3/Opus compression, re-
  encoding, y edición menor.
- **StegaStamp:** watermark en el spectrograma.
- **SynthID (Google):** propietario, integrado en
  Gemini, AudioFX, y Vertex AI.

**ASVspoof challenge.** El benchmark estándar:
ASVspoof 5 (2024) tiene 5 categorías: bona fide,
TTS, VC, audio replay, y adversarial attacks.

**Métricas.**

- **EER:** tasa de error igual. SOTA en ASVspoof 5: < 2%.
- **minDCF:** pondera FAR y FRR.
- **Robustness:** accuracy contra compresión, re-
  encoding, edición, ruido añadido.

**Aplicaciones.**

- **News/media:** verificar audios antes de publicar.
- **Banking:** verificar identidad de hablante en
  autenticación.
- **Call centers:** detectar vishing (voice phishing).
- **Compliance:** EU AI Act y Executive Orders requieren
  marcado de contenido generado por IA.

**Trampas.**

- **Falsos positivos:** un detector demasiado sensible
  bloquea audio legítimo. EER mínimo.
- **Adversarial attacks:** los generadores añaden ruido
  específicamente para evadir detectores. ASVspoof
  incluye ataques adversariales.
- **Compresión degrada:** MP3 / Opus eliminan
  artifacts que los detectores usan. Watermarks
  robustos sobreviven.

## Constrúyelo

```python
import numpy as np


def add_watermark(audio, sample_rate=24000, payload="AI-2026"):
    """Añade un watermark simple (LSB) al audio.
    En producción: AudioSeal o SynthID."""
    audio = audio.copy()
    # Convertir payload a bits
    bits = []
    for char in payload:
        for bit in format(ord(char), "08b"):
            bits.append(int(bit))
    # Insertar en LSB de cada sample
    n_bits = min(len(bits), len(audio))
    audio[:n_bits] = (
        np.floor(audio[:n_bits] * 32768).astype(np.int16) & 0xFFFE
    ) | np.array(bits[:n_bits], dtype=np.int16)
    return audio / 32768.0


def detect_synthetic_audio(audio, sr=16000):
    """Detector simple basado en features.
    En producción: RawNet2 o AudioSeal."""
    # Heurística: alta frecuencia y phase inconsistencies
    # son típicas de audio sintético
    n_fft = 1024
    spec = np.abs(np.fft.rfft(audio, n=n_fft))
    high_freq_energy = spec[len(spec) // 2:].mean()
    low_freq_energy = spec[:len(spec) // 4].mean()
    ratio = high_freq_energy / max(low_freq_energy, 1e-10)
    # Si ratio es muy alto, sospechoso
    is_synthetic = ratio > 0.5
    return is_synthetic, ratio
```

## Úsalo

```bash
pip install audioseal
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-anti-spoofing
fase: 06
leccion: 16
---

Eres un asistente que ayuda con anti-spoofing o water-
marking. Recibirás el caso de uso. Tu trabajo:

1. Para detección: RawNet2 o AASIST pre-entrenados en
   ASVspoof.
2. Para watermark en producción: AudioSeal (open-
   source) o SynthID (Google).
3. Evaluar con EER y minDCF en ASVspoof.
4. Robustness: probar con compresión MP3, ruido,
   re-encoding.
5. Para deployment: combinar detector + watermark.
6. SIEMPRE obtener consentimiento para voice cloning.
7. Compliance: EU AI Act, Executive Orders sobre
   marcado de contenido AI.
8. Advertir contra adversarial attacks.
```

## Ejercicios

1. **Detector**: implementa RawNet2 o usa un
   pre-entrenado.
2. **Watermark**: usa AudioSeal para marcar audio.
3. **Desafío**: implementa un pipeline completo
   (detector + watermark) para un dominio.

## Lecturas recomendadas

- *ASVspoof 5 Challenge* — 2024.
- *RawNet2* — Jung et al., 2020.
- *AudioSeal* — Meta, 2024.
- *SynthID* — Google, 2024.
- ASVspoof: <https://www.asvspoof.org>.

---

> 📚 **Adaptación al español** de la lección "[Anti-Spoofing and Audio Watermarking]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
